#!/usr/bin/env python3
"""
MedTokenBudget — Main experiment runner.

Evaluates lesion-preserving token routing vs. baselines across:
  - Multiple token budgets (10%, 25%, 50%, 75%, 100%)
  - 3 datasets (ISIC, BRISC, MedMNIST)
  - Fair baselines with independent heads:
    no_pruning, random, norm_based, tome, attention_entropy, local_contrast, lats

Usage:
    # Quick test on MedMNIST
    python run_experiments.py --mode quick

    # Full ISIC evaluation
    python run_experiments.py --mode full --dataset isic

    # Budget sweep
    python run_experiments.py --mode sweep --dataset isic

    # Resume interrupted training (auto-finds latest checkpoint)
    python run_experiments.py --mode full --resume auto

    # Resume from specific checkpoint
    python run_experiments.py --mode full --resume results/med_token_budget/interrupted.pt

    # All experiments
    python run_experiments.py --mode all
"""

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import timm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from config import (
    ExperimentConfig, RouterConfig, ModelConfig, DataConfig, TrainConfig,
    ISIC_BASELINE, BRISC_BASELINE, MEDMNIST_QUICK,
)
from model import MedTokenBudget
from router import LesionAwareTokenScorer, TokenRouter
from datasets import get_dataloaders
from train import MedTokenBudgetTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ─── Backbone Factory ────────────────────────────────────────────────

def get_backbone(config: ModelConfig, device: str = "cuda"):
    """Load frozen ViT backbone with graceful fallbacks."""
    backbone = None
    embed_dim = None

    if config.backbone == "dino_v2":
        try:
            backbone = torch.hub.load(
                'facebookresearch/dinov2',
                f'dinov2_vit{config.backbone_size[0]}14'
            )
            embed_dim = 768 if config.backbone_size == 'base' else 384
        except Exception as e:
            logger.warning(f"DINOv2 failed ({e}), falling back to ResNet50")

    elif config.backbone == "medmae":
        try:
            backbone = timm.create_model(
                f'vit_{config.backbone_size}_patch16_224', pretrained=True, num_classes=0)
            embed_dim = 768 if config.backbone_size == 'base' else 384
        except Exception:
            logger.warning("MedMAE failed, falling back to ResNet50")

    elif config.backbone == "sam":
        try:
            from segment_anything import sam_model_registry
            sam = sam_model_registry['vit_b'](checkpoint=None)
            backbone = sam.image_encoder
            embed_dim = 768
        except Exception:
            logger.warning("SAM not available, falling back to ResNet50")

    # Fallback: timm ViT (small, CPU-friendly, produces patch tokens)
    if backbone is None:
        logger.info("Using timm ViT-Small as fallback backbone")
        backbone = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        embed_dim = 384

    backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    return backbone, embed_dim


# ─── Baseline Implementations ────────────────────────────────────────

BASELINE_ALIASES = {
    'dynamic_vit': 'norm_based',
    'evit': 'attention_entropy',
    'freq_aware': 'local_contrast',
}


def canonical_baseline(method: str) -> str:
    return BASELINE_ALIASES.get(method, method)


def _tome_merge_once(tokens: torch.Tensor, sizes: torch.Tensor, num_merge: int):
    """One ToMe-style bipartite merge round."""
    B, N, D = tokens.shape
    if num_merge <= 0 or N <= 1:
        return tokens, sizes

    src = tokens[:, ::2]
    dst = tokens[:, 1::2]
    src_sizes = sizes[:, ::2]
    dst_sizes = sizes[:, 1::2]
    S, M = src.shape[1], dst.shape[1]
    if S == 0 or M == 0:
        return tokens, sizes

    num_merge = min(num_merge, S)
    src_norm = F.normalize(src, dim=-1)
    dst_norm = F.normalize(dst, dim=-1)
    sim = torch.bmm(src_norm, dst_norm.transpose(1, 2))
    best_sim, best_dst = sim.max(dim=-1)
    merge_src = best_sim.topk(num_merge, dim=-1, sorted=False).indices
    merge_dst = best_dst.gather(1, merge_src)

    src_values = src.gather(1, merge_src.unsqueeze(-1).expand(-1, -1, D))
    src_weights = src_sizes.gather(1, merge_src).unsqueeze(-1)
    dst_weighted = dst * dst_sizes.unsqueeze(-1)
    dst_weighted.scatter_add_(1, merge_dst.unsqueeze(-1).expand(-1, -1, D), src_values * src_weights)
    dst_sizes = dst_sizes.scatter_add(1, merge_dst, src_weights.squeeze(-1))
    dst = dst_weighted / dst_sizes.unsqueeze(-1).clamp(min=1e-6)

    keep_src = torch.ones(B, S, dtype=torch.bool, device=tokens.device)
    keep_src.scatter_(1, merge_src, False)
    kept_src = src[keep_src].view(B, S - num_merge, D)
    kept_src_sizes = src_sizes[keep_src].view(B, S - num_merge)

    merged_tokens = torch.cat([dst, kept_src], dim=1)
    merged_sizes = torch.cat([dst_sizes, kept_src_sizes], dim=1)
    return merged_tokens, merged_sizes


def tome_merging(patches: torch.Tensor, budget: float, min_tokens: int = 1) -> torch.Tensor:
    """ToMe-style: recursively merge similar patches instead of dropping tokens."""
    B, N, _ = patches.shape
    target_tokens = min(N, max(min_tokens, int(N * budget)))
    tokens = patches
    sizes = torch.ones(B, N, device=patches.device, dtype=patches.dtype)

    while tokens.shape[1] > target_tokens:
        merge_count = tokens.shape[1] - target_tokens
        tokens, sizes = _tome_merge_once(tokens, sizes, merge_count)

    return tokens


def effective_patch_size(config: ModelConfig) -> int:
    """Backbone-specific patch size used by the frozen feature extractor."""
    return 14 if config.backbone == "dino_v2" else config.patch_size


def infer_num_patches(config: ModelConfig) -> int:
    grid = max(1, config.image_size // effective_patch_size(config))
    return grid * grid


def local_contrast_scores(patches: torch.Tensor) -> torch.Tensor:
    """Training-free local contrast proxy from neighboring patch differences."""
    B, N, _ = patches.shape
    scores = torch.zeros(B, N, device=patches.device)
    if N <= 1:
        return scores
    sim = F.cosine_similarity(patches[:, :-1], patches[:, 1:], dim=-1)
    pair_score = 1.0 - sim
    counts = torch.zeros(B, N, device=patches.device)
    scores[:, :-1] += pair_score
    scores[:, 1:] += pair_score
    counts[:, :-1] += 1
    counts[:, 1:] += 1
    return scores / counts.clamp(min=1)


def attention_entropy_scores(patches: torch.Tensor, attentions: Optional[torch.Tensor]) -> torch.Tensor:
    """Attention concentration score independent of the LATS scorer."""
    B, N, _ = patches.shape
    eps = 1e-8
    if attentions is not None and attentions.dim() == 4 and attentions.shape[-1] >= N + 1:
        attn_no_cls = attentions[:, :, 1:N + 1, 1:N + 1].mean(dim=1)
        entropy = -(attn_no_cls * (attn_no_cls + eps).log()).sum(dim=-1)
    else:
        sim = F.cosine_similarity(patches.unsqueeze(2), patches.unsqueeze(1), dim=-1)
        pseudo_attn = F.softmax(sim / 0.1, dim=-1)
        entropy = -(pseudo_attn * (pseudo_attn + eps).log()).sum(dim=-1)
    return 1.0 - entropy / max(np.log(max(N, 2)), eps)


def pool_topk(
    patches: torch.Tensor,
    scores: torch.Tensor,
    budget: float,
    min_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pool selected top-K patches with the same mean-pooling contract as LATS."""
    B, N, D = patches.shape
    K = min(N, max(min_tokens, int(N * budget)))
    _, indices = scores.topk(K, dim=-1, sorted=False)
    mask = torch.zeros(B, N, device=patches.device)
    mask.scatter_(1, indices, 1.0)
    pooled = (patches * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
    return pooled, mask


def baseline_key(method: str, budget: float) -> str:
    return f"{canonical_baseline(method)}@{budget:.2f}"


def make_head_like(model: MedTokenBudget) -> torch.nn.Module:
    return copy.deepcopy(model.head)


def align_lesion_mask(lesion_masks: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if lesion_masks.dim() == 3:
        lesion_masks = lesion_masks.flatten(1)
    if lesion_masks.shape[1] > target_tokens:
        lesion_masks = lesion_masks[:, :target_tokens]
    elif lesion_masks.shape[1] < target_tokens:
        padding = torch.zeros(
            lesion_masks.shape[0],
            target_tokens - lesion_masks.shape[1],
            device=lesion_masks.device,
            dtype=lesion_masks.dtype,
        )
        lesion_masks = torch.cat([lesion_masks, padding], dim=1)
    return lesion_masks.float()


def accumulate_retention(
    selection_mask: Optional[torch.Tensor],
    lesion_masks: Optional[torch.Tensor],
    mask_valid: Optional[torch.Tensor],
    totals: Dict[str, float],
) -> None:
    if selection_mask is None or lesion_masks is None:
        return
    lesion_masks = align_lesion_mask(lesion_masks.to(selection_mask.device), selection_mask.shape[1])
    if mask_valid is not None:
        valid = mask_valid.to(selection_mask.device).bool()
    else:
        valid = lesion_masks.sum(dim=1) > 0
    if not valid.any():
        return
    valid_masks = lesion_masks[valid]
    valid_selection = selection_mask[valid]
    lesion_area = valid_masks.sum()
    if lesion_area <= 0:
        return
    totals['retained'] += float((valid_selection * valid_masks).sum().item())
    totals['lesion_area'] += float(lesion_area.item())
    totals['samples'] += int(valid.sum().item())


def baseline_pooled_tokens(
    model: MedTokenBudget,
    patches: torch.Tensor,
    attentions: Optional[torch.Tensor],
    budget: float,
    method: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    method = canonical_baseline(method)
    min_tokens = model.router.min_tokens

    if method == 'no_pruning':
        mask = torch.ones(patches.shape[:2], device=patches.device)
        return patches.mean(dim=1), mask
    if method == 'random':
        scores = torch.rand(patches.shape[:2], device=patches.device)
        return pool_topk(patches, scores, budget, min_tokens)
    if method == 'norm_based':
        scores = patches.norm(dim=-1)
        return pool_topk(patches, scores, budget, min_tokens)
    if method == 'tome':
        merged = tome_merging(patches, budget, min_tokens)
        return merged.mean(dim=1), None
    if method == 'attention_entropy':
        scores = attention_entropy_scores(patches, attentions)
        return pool_topk(patches, scores, budget, min_tokens)
    if method == 'local_contrast':
        scores = local_contrast_scores(patches)
        return pool_topk(patches, scores, budget, min_tokens)
    raise ValueError(f"Unknown baseline: {method}")


def unpack_batch(batch):
    if len(batch) >= 4:
        images, labels, lesion_masks, mask_valid = batch[:4]
    elif len(batch) == 3:
        images, labels, lesion_masks = batch
        mask_valid = None
    else:
        images, labels = batch
        lesion_masks = None
        mask_valid = None
    return images, labels, lesion_masks, mask_valid


def train_baseline_head(
    model: MedTokenBudget,
    train_loader,
    val_loader,
    device: str,
    method: str,
    budget: float,
    epochs: int,
) -> torch.nn.Module:
    """Train a fair independent head for a frozen baseline token distribution."""
    method = canonical_baseline(method)
    head = make_head_like(model).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
    best_state = copy.deepcopy(head.state_dict())
    best_acc = -1.0

    model.eval()
    for epoch in range(epochs):
        head.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            images, labels, _, _ = unpack_batch(batch)
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                patches, attentions = model.extract_patches(images)
                pooled, _ = baseline_pooled_tokens(model, patches, attentions, budget, method)
            logits = head(pooled)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += labels.numel()

        metrics = evaluate_selection_baseline(model, val_loader, device, budget, method, head)
        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            best_state = copy.deepcopy(head.state_dict())
        logger.info(
            f"  train head {baseline_key(method, budget)} epoch {epoch + 1}/{epochs} | "
            f"loss={total_loss / max(len(train_loader), 1):.4f} | "
            f"train_acc={correct / max(total, 1):.4f} | val_acc={metrics['accuracy']:.4f}"
        )

    head.load_state_dict(best_state)
    return head


@torch.no_grad()
def evaluate_selection_baseline(
    model: MedTokenBudget,
    val_loader,
    device: str,
    budget: float,
    method: str,
    head: torch.nn.Module,
) -> Dict[str, float]:
    """Evaluate a token selection baseline with its own independently trained head."""
    model.eval()
    head.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    retention = {'retained': 0.0, 'lesion_area': 0.0, 'samples': 0}

    for batch in val_loader:
        images, labels, lesion_masks, mask_valid = unpack_batch(batch)
        images = images.to(device)
        labels = labels.to(device)
        patches, attentions = model.extract_patches(images)
        pooled, selection_mask = baseline_pooled_tokens(model, patches, attentions, budget, method)
        if lesion_masks is not None:
            accumulate_retention(selection_mask, lesion_masks, mask_valid, retention)

        logits = head(pooled)
        total_loss += F.cross_entropy(logits, labels).item()
        all_preds.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).argmax(-1).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics = {
        'loss': total_loss / len(val_loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'macro_f1': f1_score(all_labels, all_preds, average='macro'),
        'weighted_f1': f1_score(all_labels, all_preds, average='weighted'),
    }
    if retention['lesion_area'] > 0:
        metrics['lesion_retention'] = retention['retained'] / retention['lesion_area']
        metrics['retention_samples'] = retention['samples']
    return metrics


@torch.no_grad()
def evaluate_lats_retention(
    model: MedTokenBudget,
    val_loader,
    device: str,
    budget: float,
) -> Dict[str, float]:
    """Compute lesion-mask retention for LATS when masks are available."""
    model.eval()
    retention = {'retained': 0.0, 'lesion_area': 0.0, 'samples': 0}
    for batch in val_loader:
        images, _, lesion_masks, mask_valid = unpack_batch(batch)
        if lesion_masks is None:
            continue
        images = images.to(device)
        output = model(images, budget_ratio=budget, return_routing_info=True)
        accumulate_retention(output['selection_mask'], lesion_masks, mask_valid, retention)

    if retention['lesion_area'] <= 0:
        return {}
    return {
        'lesion_retention': retention['retained'] / retention['lesion_area'],
        'retention_samples': retention['samples'],
    }


# ─── Budget Sweep ────────────────────────────────────────────────────

def run_budget_sweep(
    model: MedTokenBudget,
    train_loader,
    val_loader,
    trainer: MedTokenBudgetTrainer,
    config: ExperimentConfig,
    budgets: List[float] = [0.1, 0.25, 0.5, 0.75, 1.0],
    baselines: List[str] = None,
):
    """Evaluate accuracy across token budgets with fair independently trained heads."""
    if baselines is None:
        baselines = ['no_pruning', 'random', 'norm_based', 'tome', 'attention_entropy', 'local_contrast', 'lats']

    baselines = [canonical_baseline(b) for b in baselines]
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    head_path = output_dir / 'baseline_heads.pt'
    baseline_head_states: Dict[str, Dict[str, torch.Tensor]] = {}

    if head_path.exists():
        try:
            cached = torch.load(head_path, map_location='cpu', weights_only=False)
            baseline_head_states = cached.get('heads', cached)
            logger.info(f"Loaded cached baseline heads: {head_path}")
        except Exception as exc:
            logger.warning(f"Could not load cached baseline heads; retraining: {exc}")
            baseline_head_states = {}

    budget_for_head = lambda method, budget: 1.0 if method == 'no_pruning' else budget

    for budget in budgets:
        for baseline in baselines:
            if baseline == 'lats':
                continue
            train_budget = budget_for_head(baseline, budget)
            key = baseline_key(baseline, train_budget)
            if key in baseline_head_states:
                continue
            logger.info(f"Training independent baseline head: {key}")
            head = train_baseline_head(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=trainer.device,
                method=baseline,
                budget=train_budget,
                epochs=getattr(config, 'baseline_head_epochs', 5),
            )
            baseline_head_states[key] = {
                name: value.detach().cpu()
                for name, value in head.state_dict().items()
            }
            torch.save({'heads': baseline_head_states}, head_path)

    results = {b: {} for b in budgets}

    for budget in budgets:
        logger.info(f"Evaluating budget={budget}...")

        # LATS
        model.set_budget(budget)
        lats_metrics = trainer.validate(val_loader, budget=budget)
        lats_metrics.update(evaluate_lats_retention(model, val_loader, trainer.device, budget))
        results[budget]['lats'] = lats_metrics

        for baseline in baselines:
            if baseline == 'lats':
                continue
            eval_budget = budget_for_head(baseline, budget)
            key = baseline_key(baseline, eval_budget)
            head = make_head_like(model).to(trainer.device)
            head.load_state_dict(baseline_head_states[key])
            baseline_metrics = evaluate_selection_baseline(
                model, val_loader, trainer.device, eval_budget, baseline, head
            )
            results[budget][baseline] = baseline_metrics
            logger.info(f"  {baseline} Acc: {baseline_metrics['accuracy']:.4f} | "
                        f"F1: {baseline_metrics['macro_f1']:.4f}")

        logger.info(f"  LATS Acc: {lats_metrics['accuracy']:.4f}")

    return results


def run_signal_ablation(
    model: MedTokenBudget,
    val_loader,
    trainer: MedTokenBudgetTrainer,
    budget: float = 0.25,
) -> Dict[str, Dict[str, float]]:
    """Inference ablation for LATS scorer signals using the trained scorer/head."""
    settings = {
        'all': (True, True, True),
        'no_attention': (False, True, True),
        'no_norm': (True, False, True),
        'no_local_contrast': (True, True, False),
        'attention_only': (True, False, False),
        'norm_only': (False, True, False),
        'local_contrast_only': (False, False, True),
    }
    scorer = model.scorer
    original = (
        scorer.use_attention_entropy,
        scorer.use_feature_norm,
        scorer.use_frequency_content,
    )
    results = {}
    try:
        for name, (use_attn, use_norm, use_freq) in settings.items():
            scorer.use_attention_entropy = use_attn
            scorer.use_feature_norm = use_norm
            scorer.use_frequency_content = use_freq
            metrics = trainer.validate(val_loader, budget=budget)
            metrics.update(evaluate_lats_retention(model, val_loader, trainer.device, budget))
            results[name] = metrics
            logger.info(
                f"  ablation {name}: Acc={metrics['accuracy']:.4f} | "
                f"F1={metrics['macro_f1']:.4f}"
            )
    finally:
        (
            scorer.use_attention_entropy,
            scorer.use_feature_norm,
            scorer.use_frequency_content,
        ) = original
    return results


def find_eval_checkpoint(output_dir: str) -> Optional[str]:
    """Find a trained checkpoint for evaluation-only modes."""
    output_dir = Path(output_dir)
    for name in ['best_model.pt', 'latest.pt', 'final_model.pt']:
        path = output_dir / name
        if path.exists():
            return str(path)
    autos = sorted(output_dir.glob('auto_epoch_*.pt'), key=lambda x: x.stat().st_mtime, reverse=True)
    return str(autos[0]) if autos else None


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MedTokenBudget Experiments")
    parser.add_argument('--mode', choices=['quick', 'full', 'sweep', 'ablation', 'all'],
                       default='quick')
    parser.add_argument('--dataset', choices=['medmnist', 'isic', 'brisc'],
                       default='medmnist')
    parser.add_argument('--output_dir', default='./results/med_token_budget')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint ("auto" = find latest, or path to .pt)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    # Select config
    if args.mode == 'quick':
        config = MEDMNIST_QUICK
        config.data.medmnist_subset = 'pathmnist'
        config.train.epochs = 10
    elif args.dataset == 'isic':
        config = ISIC_BASELINE
    elif args.dataset == 'brisc':
        config = BRISC_BASELINE
    else:
        config = MEDMNIST_QUICK
        config.data.medmnist_subset = args.dataset if args.dataset != 'medmnist' else 'pathmnist'

    config.output_dir = args.output_dir
    config.device = device
    config.seed = args.seed

    # Load data
    logger.info(f"Loading {config.data.dataset} dataset...")
    dataloaders = get_dataloaders(config)
    config.model.num_classes = dataloaders['num_classes']
    logger.info(f"Train: {dataloaders['train_size']}, Val: {dataloaders['val_size']}, "
               f"Classes: {dataloaders['num_classes']}")

    # Load backbone
    logger.info(f"Loading {config.model.backbone} backbone...")
    backbone, embed_dim = get_backbone(config.model, device)

    # Create model
    router_cfg = {
        'hidden_dim': config.router.router_hidden_dim,
        'num_layers': config.router.router_num_layers,
        'dropout': config.router.router_dropout,
        'use_attention_entropy': config.router.use_attention_entropy,
        'use_feature_norm': config.router.use_feature_norm,
        'use_frequency_content': config.router.use_frequency_content,
        'token_budget_ratio': config.router.token_budget_ratio,
        'min_tokens': config.router.min_tokens,
        'spatial_smoothing_kernel': config.router.spatial_smoothing_kernel,
        'temperature': config.router.temperature,
    }
    head_cfg = {
        'type': config.model.head_type,
        'hidden_dim': config.model.head_hidden_dim,
    }

    model = MedTokenBudget(
        backbone=backbone,
        embed_dim=embed_dim,
        num_classes=config.model.num_classes,
        num_patches=infer_num_patches(config.model),
        router_config=router_cfg,
        head_config=head_cfg,
    ).to(device)

    trainable_count = sum(p.numel() for p in model.get_trainable_params())
    total_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_count:,} total params, {trainable_count:,} trainable")

    # Train
    trainer = MedTokenBudgetTrainer(model, config, device)

    if args.mode in ['sweep', 'ablation']:
        resume_path = args.resume
        if resume_path is None or resume_path == 'auto':
            resume_path = find_eval_checkpoint(args.output_dir)
        if not resume_path or not Path(resume_path).exists():
            raise FileNotFoundError(
                f"{args.mode} requires a trained checkpoint in {args.output_dir}. "
                "Pass --resume PATH or upload best_model.pt/latest.pt."
            )
        logger.info(f"Loading checkpoint for {args.mode}: {resume_path}")
        trainer.load_checkpoint(resume_path)

    if args.mode in ['quick', 'full', 'all']:
        # Check for resume
        resume_path = None
        if args.resume == 'auto':
            resume_path = trainer.find_latest_checkpoint()
            if resume_path:
                logger.info(f"🔍 Found checkpoint: {resume_path}")
            else:
                logger.info("No checkpoint found — starting fresh.")
        elif args.resume:
            resume_path = args.resume

        if resume_path and Path(resume_path).exists():
            logger.info(f"🔄 Resuming from: {resume_path}")
            trainer.resume(resume_path, dataloaders['train'], dataloaders['val'])
        else:
            logger.info("Starting training...")
            trainer.train(dataloaders['train'], dataloaders['val'])

        trainer.save_checkpoint('final_model.pt')

        best_path = Path(args.output_dir) / 'best_model.pt'
        if best_path.exists():
            logger.info(f"Loading best checkpoint for evaluation: {best_path}")
            trainer.load_checkpoint(str(best_path))

    # Budget sweep
    if args.mode in ['sweep', 'all']:
        logger.info("Running budget sweep...")
        budgets = [0.1, 0.25, 0.5, 0.75, 1.0]
        sweep_results = run_budget_sweep(
            model,
            dataloaders['train'],
            dataloaders['val'],
            trainer,
            config,
            budgets,
            config.baselines,
        )

        # Save
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'budget_sweep.json', 'w') as f:
            json.dump(sweep_results, f, indent=2, default=str)
        logger.info(f"Sweep results saved to {output_dir / 'budget_sweep.json'}")

    if args.mode in ['ablation', 'all']:
        logger.info("Running LATS signal ablation...")
        ablation_results = run_signal_ablation(
            model, dataloaders['val'], trainer, budget=0.25
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'signal_ablation.json', 'w') as f:
            json.dump(ablation_results, f, indent=2, default=str)
        logger.info(f"Signal ablation saved to {output_dir / 'signal_ablation.json'}")

    logger.info("Done!")


if __name__ == '__main__':
    main()

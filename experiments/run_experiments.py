#!/usr/bin/env python3
"""
MedTokenBudget — Main experiment runner.

Evaluates lesion-preserving token routing vs. baselines across:
  - Multiple token budgets (10%, 25%, 50%, 75%, 100%)
  - 3 datasets (ISIC, BRISC, MedMNIST)
  - 5 baselines (no_pruning, random, dynamic_vit, tome, evit, lats)

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
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

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

def random_token_pruning(patches: torch.Tensor, budget: float) -> torch.Tensor:
    """Randomly select patches."""
    B, N, D = patches.shape
    K = max(1, int(N * budget))
    indices = torch.randperm(N)[:K]
    return patches[:, indices, :]


def dynamic_vit_pruning(patches: torch.Tensor, budget: float) -> torch.Tensor:
    """DynamicViT-style: prune based on attention scores from CLS."""
    # Simplified: use feature norm as proxy for attention
    B, N, D = patches.shape
    K = max(1, int(N * budget))
    scores = patches.norm(dim=-1)
    _, indices = scores.topk(K, dim=-1)
    selected = torch.gather(patches, 1,
                           indices.unsqueeze(-1).expand(-1, -1, D))
    return selected


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


def frequency_aware_scores(patches: torch.Tensor) -> torch.Tensor:
    """Training-free high-frequency proxy from neighboring patch differences."""
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


def pool_topk(patches: torch.Tensor, scores: torch.Tensor, budget: float, min_tokens: int) -> torch.Tensor:
    """Pool selected top-K patches with the same mean-pooling contract as LATS."""
    B, N, D = patches.shape
    K = min(N, max(min_tokens, int(N * budget)))
    _, indices = scores.topk(K, dim=-1, sorted=False)
    mask = torch.zeros(B, N, device=patches.device)
    mask.scatter_(1, indices, 1.0)
    return (patches * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)


@torch.no_grad()
def evaluate_selection_baseline(
    model: MedTokenBudget,
    val_loader,
    device: str,
    budget: float,
    method: str,
) -> Dict[str, float]:
    """Evaluate training-free token selection baselines with the trained head."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    min_tokens = model.router.min_tokens

    for batch in val_loader:
        images, labels = batch[:2]
        images = images.to(device)
        labels = labels.to(device)
        patches, attentions = model.extract_patches(images)

        if method == 'no_pruning':
            pooled = patches.mean(dim=1)
        elif method == 'random':
            scores = torch.rand(patches.shape[:2], device=device)
            pooled = pool_topk(patches, scores, budget, min_tokens)
        elif method == 'dynamic_vit':
            scores = patches.norm(dim=-1)
            pooled = pool_topk(patches, scores, budget, min_tokens)
        elif method == 'tome':
            merged = tome_merging(patches, budget, min_tokens)
            pooled = merged.mean(dim=1)
        elif method == 'evit':
            signals = model.scorer.compute_signals(patches, attentions)
            scores = signals['attention'].squeeze(-1)
            pooled = pool_topk(patches, scores, budget, min_tokens)
        elif method == 'freq_aware':
            scores = frequency_aware_scores(patches)
            pooled = pool_topk(patches, scores, budget, min_tokens)
        else:
            raise ValueError(f"Unknown baseline: {method}")

        logits = model.head(pooled)
        total_loss += F.cross_entropy(logits, labels).item()
        all_preds.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).argmax(-1).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return {
        'loss': total_loss / len(val_loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'macro_f1': f1_score(all_labels, all_preds, average='macro'),
        'weighted_f1': f1_score(all_labels, all_preds, average='weighted'),
    }


# ─── Budget Sweep ────────────────────────────────────────────────────

def run_budget_sweep(
    model: MedTokenBudget,
    val_loader,
    trainer: MedTokenBudgetTrainer,
    budgets: List[float] = [0.1, 0.25, 0.5, 0.75, 1.0],
    baselines: List[str] = None,
):
    """Evaluate accuracy across token budgets."""
    if baselines is None:
        baselines = ['no_pruning', 'random', 'dynamic_vit', 'tome', 'evit', 'freq_aware', 'lats']

    results = {b: {} for b in budgets}

    for budget in budgets:
        logger.info(f"Evaluating budget={budget}...")

        # LATS
        model.set_budget(budget)
        lats_metrics = trainer.validate(val_loader, budget=budget)
        results[budget]['lats'] = lats_metrics

        for baseline in baselines:
            if baseline == 'lats':
                continue
            baseline_metrics = evaluate_selection_baseline(
                model, val_loader, trainer.device, budget, baseline
            )
            results[budget][baseline] = baseline_metrics
            logger.info(f"  {baseline} Acc: {baseline_metrics['accuracy']:.4f} | "
                        f"F1: {baseline_metrics['macro_f1']:.4f}")

        logger.info(f"  LATS Acc: {lats_metrics['accuracy']:.4f}")

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
    parser.add_argument('--mode', choices=['quick', 'full', 'sweep', 'all'],
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
        router_config=router_cfg,
        head_config=head_cfg,
    ).to(device)

    trainable_count = sum(p.numel() for p in model.get_trainable_params())
    total_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_count:,} total params, {trainable_count:,} trainable")

    # Train
    trainer = MedTokenBudgetTrainer(model, config, device)

    if args.mode == 'sweep':
        resume_path = args.resume
        if resume_path is None or resume_path == 'auto':
            resume_path = find_eval_checkpoint(args.output_dir)
        if not resume_path or not Path(resume_path).exists():
            raise FileNotFoundError(
                f"Sweep requires a trained checkpoint in {args.output_dir}. "
                "Pass --resume PATH or upload best_model.pt/latest.pt."
            )
        logger.info(f"Loading checkpoint for sweep: {resume_path}")
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

    # Budget sweep
    if args.mode in ['sweep', 'all']:
        logger.info("Running budget sweep...")
        budgets = [0.1, 0.25, 0.5, 0.75, 1.0]
        sweep_results = run_budget_sweep(
            model, dataloaders['val'], trainer, budgets
        )

        # Save
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'budget_sweep.json', 'w') as f:
            json.dump(sweep_results, f, indent=2, default=str)
        logger.info(f"Sweep results saved to {output_dir / 'budget_sweep.json'}")

    logger.info("Done!")


if __name__ == '__main__':
    main()

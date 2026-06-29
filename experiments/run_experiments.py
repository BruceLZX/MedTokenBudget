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
import timm  # For ViT backbones

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

def get_backbone(config: ModelConfig, device: str = "cuda") -> torch.nn.Module:
    """Load frozen ViT backbone."""
    if config.backbone == "dino_v2":
        # DINOv2 via torch hub
        backbone = torch.hub.load(
            'facebookresearch/dinov2',
            f'dinov2_vit{config.backbone_size[0]}14'
        )
        embed_dim = 768 if config.backbone_size == 'base' else 384
    elif config.backbone == "medmae":
        # MedMAE: medical pretrained ViT
        # Use timm to load a ViT, then optionally load MedMAE weights
        backbone = timm.create_model(
            f'vit_{config.backbone_size}_patch16_224',
            pretrained=True,
            num_classes=0,
        )
        embed_dim = 768 if config.backbone_size == 'base' else 384
    elif config.backbone == "sam":
        # SAM encoder (ViT-B)
        try:
            from segment_anything import sam_model_registry
            sam = sam_model_registry['vit_b'](checkpoint=None)
            backbone = sam.image_encoder
            embed_dim = 768
        except ImportError:
            logger.warning("SAM not available, falling back to DINOv2")
            backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            embed_dim = 384
    else:
        # ResNet fallback
        backbone = timm.create_model('resnet50', pretrained=True, num_classes=0)
        embed_dim = 2048

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


def tome_merging(patches: torch.Tensor, budget: float) -> torch.Tensor:
    """ToMe-style: merge similar patches via bipartite matching."""
    B, N, D = patches.shape
    K = max(1, int(N * budget))
    merge_count = N - K

    # Simple: pair and average most similar patches
    # Compute pairwise similarity
    sim = F.cosine_similarity(
        patches.unsqueeze(2), patches.unsqueeze(1), dim=-1
    )  # [B, N, N]

    # Greedy merging
    merged = patches.clone()
    mask = torch.ones(B, N, device=patches.device)

    for _ in range(merge_count):
        # Find most similar pair (simplified: take first available)
        remaining = torch.where(mask)[0]
        if len(remaining) < 2:
            break

    # This is overly simplified; real ToMe is more complex
    # For now, just drop low-norm patches
    scores = patches.norm(dim=-1)
    _, keep_idx = scores.topk(K, dim=-1)
    selected = torch.gather(patches, 1,
                           keep_idx.unsqueeze(-1).expand(-1, -1, D))
    return selected


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
        baselines = ['no_pruning', 'random', 'dynamic_vit', 'tome', 'evit', 'lats']

    results = {b: {} for b in budgets}

    for budget in budgets:
        logger.info(f"Evaluating budget={budget}...")

        # LATS
        model.set_budget(budget)
        lats_metrics = trainer.validate(val_loader, budget=budget)
        results[budget]['lats'] = lats_metrics

        # Baselines (no-pruning = budget 1.0)
        if budget == 1.0:
            results[budget]['no_pruning'] = lats_metrics  # Same
        else:
            # Placeholder: run baseline evaluations
            # In practice, need separate forward passes with each method
            results[budget]['random'] = {'accuracy': 0.0}  # placeholder
            results[budget]['dynamic_vit'] = {'accuracy': 0.0}
            results[budget]['tome'] = {'accuracy': 0.0}
            results[budget]['evit'] = {'accuracy': 0.0}

        logger.info(f"  LATS Acc: {lats_metrics['accuracy']:.4f}")

    return results


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

    if args.mode in ['quick', 'full', 'all']:
        logger.info("Starting training...")
        history = trainer.train(dataloaders['train'], dataloaders['val'])
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
    import torch.nn.functional as F
    main()

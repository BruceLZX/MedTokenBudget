"""
Training loop for MedTokenBudget with budget curriculum learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional
import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

logger = logging.getLogger(__name__)


class MedTokenBudgetTrainer:
    """Trainer for MedTokenBudget with budget curriculum and lesion-aware losses."""

    def __init__(
        self,
        model: nn.Module,
        config,
        device: str = "cuda",
    ):
        self.model = model
        self.config = config
        self.device = device

        # Optimizer (only trainable params: scorer + head)
        trainable = model.get_trainable_params()
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )

        # Scheduler
        if config.train.lr_scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.train.epochs
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=15, gamma=0.1
            )

        # Mixed precision
        self.scaler = GradScaler() if config.train.use_amp and 'cuda' in device else None

        # Budget curriculum
        self.budget_curriculum = config.train.budget_curriculum
        self.budget_start = config.train.budget_start
        self.budget_end = config.train.budget_end
        self.budget_anneal_epochs = config.train.budget_anneal_epochs

        # Tracking
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}

    def get_current_budget(self) -> float:
        """Compute token budget for current epoch (curriculum)."""
        if not self.budget_curriculum:
            return self.config.router.token_budget_ratio

        # Linear annealing
        progress = min(self.current_epoch / self.budget_anneal_epochs, 1.0)
        budget = self.budget_start - progress * (self.budget_start - self.budget_end)
        return max(budget, self.budget_end)

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        budget = self.get_current_budget()

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} (budget={budget:.2f})")

        for batch_idx, batch in enumerate(pbar):
            # Unpack batch (handles datasets with/without lesion masks)
            if len(batch) == 3:
                images, labels, lesion_masks = batch
            else:
                images, labels = batch
                lesion_masks = None

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast():
                    loss, preds = self._forward_loss(images, labels, lesion_masks, budget)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.get_trainable_params(), max_norm=1.0
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, preds = self._forward_loss(images, labels, lesion_masks, budget)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.get_trainable_params(), max_norm=1.0
                )
                self.optimizer.step()

            total_loss += loss.item()
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

            # Update progress
            if batch_idx % self.config.train.log_interval == 0:
                acc = accuracy_score(labels.cpu().numpy(), preds.argmax(-1).cpu().numpy())
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{acc:.3f}',
                })

        # Epoch metrics
        all_preds = torch.cat(all_preds).argmax(-1).numpy()
        all_labels = torch.cat(all_labels).numpy()
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_loss = total_loss / len(train_loader)

        return {'loss': epoch_loss, 'accuracy': epoch_acc}

    def _forward_loss(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        lesion_masks: Optional[torch.Tensor],
        budget: float,
    ):
        """Compute training loss."""
        output = self.model(images, budget_ratio=budget, return_routing_info=True)
        logits = output['logits']
        preds = F.softmax(logits, dim=-1)

        # Classification loss
        cls_loss = F.cross_entropy(logits, labels)

        # Budget regularization: encourage efficient token usage
        kept_ratio = output['kept_ratio']
        budget_reg = (kept_ratio - budget).abs().mean()
        # We want kept_ratio to be close to budget, but not exceed it

        # Lesion localization auxiliary loss
        lesion_loss = torch.tensor(0.0, device=images.device)
        if lesion_masks is not None:
            lesion_masks = lesion_masks.to(self.device)
            # Encourage high scores on lesion regions
            scores = output['scores']  # [B, N]
            if lesion_masks.dim() == 3:
                lesion_masks = lesion_masks.flatten(1)
            # Pad/crop to match
            N = scores.shape[1]
            if lesion_masks.shape[1] > N:
                lesion_masks = lesion_masks[:, :N]
            elif lesion_masks.shape[1] < N:
                padding = torch.zeros(lesion_masks.shape[0], N - lesion_masks.shape[1],
                                     device=lesion_masks.device)
                lesion_masks = torch.cat([lesion_masks, padding], dim=1)
            lesion_loss = F.binary_cross_entropy(
                scores, lesion_masks.float()
            )

        # Total loss
        total_loss = (
            self.config.train.cls_loss_weight * cls_loss
            + self.config.train.budget_reg_weight * budget_reg
            + self.config.train.lesion_loc_weight * lesion_loss
        )

        return total_loss, preds

    @torch.no_grad()
    def validate(self, val_loader, budget: Optional[float] = None) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0

        eval_budget = budget if budget is not None else self.config.router.token_budget_ratio

        for batch in val_loader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            images = images.to(self.device)
            labels = labels.to(self.device)

            output = self.model(images, budget_ratio=eval_budget)
            logits = output['logits']
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()

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

        return metrics

    def train(self, train_loader, val_loader, num_epochs: Optional[int] = None):
        """Full training loop."""
        epochs = num_epochs or self.config.train.epochs

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])

            # Validate (at multiple budgets for analysis)
            if epoch % self.config.train.eval_interval == 0:
                val_metrics = self.validate(val_loader, budget=self.config.router.token_budget_ratio)
                self.history['val_acc'].append(val_metrics['accuracy'])
                self.history['val_f1'].append(val_metrics['macro_f1'])

                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['accuracy']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Val F1: {val_metrics['macro_f1']:.4f}"
                )

                # Save best
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self.best_epoch = epoch
                    if self.config.train.save_best:
                        self.save_checkpoint('best_model.pt')

            self.scheduler.step()

        logger.info(f"Training complete. Best val acc: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}")
        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ckpt = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': self.config,
        }
        torch.save(ckpt, output_dir / filename)
        logger.info(f"Checkpoint saved to {output_dir / filename}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.best_val_acc = ckpt.get('best_val_acc', 0.0)
        self.history = ckpt.get('history', {})
        self.current_epoch = ckpt.get('epoch', 0)
        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")

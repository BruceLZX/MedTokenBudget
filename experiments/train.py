"""
Training loop for MedTokenBudget with budget curriculum + checkpoint resume.

Supports Ctrl+C interrupt → resume from exact epoch.
Auto-saves every 5 epochs + keeps best model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional
import logging
import signal
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

logger = logging.getLogger(__name__)


class MedTokenBudgetTrainer:
    """Trainer with budget curriculum, lesion-aware losses, and checkpoint resume."""

    def __init__(self, model: nn.Module, config, device: str = "cuda"):
        self.model = model
        self.config = config
        self.device = device

        # Trainable params
        trainable = model.get_trainable_params()
        self.optimizer = torch.optim.AdamW(
            trainable, lr=config.train.lr, weight_decay=config.train.weight_decay)

        if config.train.lr_scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.train.epochs)
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=15, gamma=0.1)

        self.scaler = GradScaler() if config.train.use_amp and 'cuda' in device else None

        # Budget curriculum
        self.budget_curriculum = config.train.budget_curriculum
        self.budget_start = config.train.budget_start
        self.budget_end = config.train.budget_end
        self.budget_anneal_epochs = config.train.budget_anneal_epochs

        # State tracking
        self.current_epoch = 0
        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': [],
                        'epoch': [], 'lr': []}
        self._interrupted = False

        # Register signal handler for graceful interrupt
        self._original_sigint = signal.getsignal(signal.SIGINT)

    def _handle_interrupt(self, signum, frame):
        """Graceful interrupt: save checkpoint before exiting."""
        logger.info("\n⚠️  Interrupted! Saving checkpoint...")
        self._interrupted = True
        self.save_checkpoint('interrupted.pt')
        logger.info("Checkpoint saved. Resume with: --resume interrupted.pt")
        # Restore original handler and re-raise
        signal.signal(signal.SIGINT, self._original_sigint)
        raise KeyboardInterrupt

    def get_current_budget(self) -> float:
        """Compute token budget for current epoch (curriculum)."""
        if not self.budget_curriculum:
            return self.config.router.token_budget_ratio
        progress = min(self.current_epoch / max(self.budget_anneal_epochs, 1), 1.0)
        if getattr(self.config.train, 'budget_schedule', 'linear') == 'cosine':
            progress = 0.5 * (1.0 - np.cos(np.pi * progress))
        return self.budget_start - progress * (self.budget_start - self.budget_end)

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        budget = self.get_current_budget()
        score_means, score_stds = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} (budget={budget:.2f})")

        for batch_idx, batch in enumerate(pbar):
            if len(batch) >= 4:
                images, labels, lesion_masks, mask_valid = batch[:4]
            elif len(batch) == 3:
                images, labels, lesion_masks = batch
                mask_valid = None
            else:
                images, labels = batch
                lesion_masks = None
                mask_valid = None

            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast():
                    loss, preds = self._forward_loss(images, labels, lesion_masks, mask_valid, budget)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch {self.current_epoch + 1}, batch {batch_idx}: {loss.item()}")
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.get_trainable_params(), max_norm=1.0)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(
                        f"Non-finite gradient norm at epoch {self.current_epoch + 1}, batch {batch_idx}: {grad_norm.item()}"
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, preds = self._forward_loss(images, labels, lesion_masks, mask_valid, budget)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch {self.current_epoch + 1}, batch {batch_idx}: {loss.item()}")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.get_trainable_params(), max_norm=1.0)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(
                        f"Non-finite gradient norm at epoch {self.current_epoch + 1}, batch {batch_idx}: {grad_norm.item()}"
                    )
                self.optimizer.step()

            total_loss += loss.item()
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())
            if hasattr(self, '_last_score_stats'):
                mean, std = self._last_score_stats
                score_means.append(mean)
                score_stds.append(std)

            if batch_idx % self.config.train.log_interval == 0:
                acc = accuracy_score(labels.cpu().numpy(), preds.argmax(-1).cpu().numpy())
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.3f}'})

        all_preds = torch.cat(all_preds).argmax(-1).numpy()
        all_labels = torch.cat(all_labels).numpy()
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'score_mean': float(np.mean(score_means)) if score_means else 0.0,
            'score_std': float(np.mean(score_stds)) if score_stds else 0.0,
        }

    def _forward_loss(self, images, labels, lesion_masks, mask_valid, budget):
        output = self.model(images, budget_ratio=budget, return_routing_info=True)
        logits = output['logits']
        preds = F.softmax(logits, dim=-1)
        self._last_score_stats = (
            output['scores'].detach().float().mean().item(),
            output['scores'].detach().float().std().item(),
        )

        cls_loss = F.cross_entropy(logits, labels)
        kept_ratio = output['kept_ratio']
        budget_reg = (kept_ratio - budget).abs().mean()

        lesion_loss = torch.tensor(0.0, device=images.device)
        if lesion_masks is not None:
            lesion_masks = lesion_masks.to(self.device)
            if mask_valid is not None:
                mask_valid = mask_valid.to(self.device).bool()
            scores = output['scores']
            if lesion_masks.dim() == 3:
                lesion_masks = lesion_masks.flatten(1)
            N = scores.shape[1]
            if lesion_masks.shape[1] > N:
                lesion_masks = lesion_masks[:, :N]
            elif lesion_masks.shape[1] < N:
                padding = torch.zeros(lesion_masks.shape[0], N - lesion_masks.shape[1],
                                     device=lesion_masks.device)
                lesion_masks = torch.cat([lesion_masks, padding], dim=1)
            if mask_valid is None or mask_valid.any():
                valid_scores = scores if mask_valid is None else scores[mask_valid]
                valid_masks = lesion_masks.float() if mask_valid is None else lesion_masks[mask_valid].float()
                lesion_loss = F.binary_cross_entropy(valid_scores, valid_masks)

        # Diversity loss: penalize if all images in batch get similar score patterns
        div_loss = torch.tensor(0.0, device=images.device)
        B = images.shape[0]
        scores_flat = output['scores']  # [B, N]
        if self.config.train.diversity_weight > 0 and B > 1:
            score_norm = F.normalize(scores_flat, dim=-1)
            pairwise_sim = (score_norm @ score_norm.T).clamp(min=0)
            off_diag = pairwise_sim[~torch.eye(B, dtype=torch.bool, device=images.device)]
            div_loss = off_diag.mean()

        # Attention distillation: use frozen-backbone CLS attention when available.
        attn_loss = torch.tensor(0.0, device=images.device)
        if self.config.train.attention_distill_weight > 0:
            attentions = output.get('attentions')
            if attentions is not None:
                with torch.no_grad():
                    teacher = attentions[:, :, 0, 1:].mean(dim=1)
                    teacher = teacher[:, :scores_flat.shape[1]]
                    teacher = teacher / teacher.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
                attn_loss = F.mse_loss(scores_flat, teacher)

        total_loss = (self.config.train.cls_loss_weight * cls_loss
                      + self.config.train.budget_reg_weight * budget_reg
                      + self.config.train.lesion_loc_weight * lesion_loss
                      + self.config.train.diversity_weight * div_loss
                      + self.config.train.attention_distill_weight * attn_loss)
        return total_loss, preds

    @torch.no_grad()
    def validate(self, val_loader, budget: Optional[float] = None) -> Dict[str, float]:
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0
        eval_budget = budget if budget is not None else self.config.router.token_budget_ratio

        for batch in val_loader:
            if len(batch) >= 3:
                images, labels = batch[:2]
            else:
                images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            output = self.model(images, budget_ratio=eval_budget)
            logits = output['logits']
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

    def train(self, train_loader, val_loader, num_epochs: Optional[int] = None):
        """Full training loop with checkpoint resume support."""
        epochs = num_epochs or self.config.train.epochs

        signal.signal(signal.SIGINT, self._handle_interrupt)

        try:
            for epoch in range(self.start_epoch, epochs):
                self.current_epoch = epoch

                train_metrics = self.train_epoch(train_loader)
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['train_acc'].append(train_metrics['accuracy'])

                if epoch % self.config.train.eval_interval == 0 or epoch == epochs - 1:
                    val_metrics = self.validate(val_loader)
                    self.history['val_acc'].append(val_metrics['accuracy'])
                    self.history['val_f1'].append(val_metrics['macro_f1'])
                    self.history['epoch'].append(epoch)
                    self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

                    logger.info(
                        f"Epoch {epoch+1}/{epochs} | "
                        f"Train Loss: {train_metrics['loss']:.4f} | "
                        f"Train Acc: {train_metrics['accuracy']:.4f} | "
                        f"Score μ/σ: {train_metrics.get('score_mean', 0.0):.3f}/"
                        f"{train_metrics.get('score_std', 0.0):.3f} | "
                        f"Val Acc: {val_metrics['accuracy']:.4f} | "
                        f"Val F1: {val_metrics['macro_f1']:.4f}"
                    )

                    min_delta = getattr(self.config.train, 'early_stopping_min_delta', 0.0)
                    if val_metrics['accuracy'] > self.best_val_acc + min_delta:
                        self.best_val_acc = val_metrics['accuracy']
                        self.best_epoch = epoch
                        if self.config.train.save_best:
                            self.save_checkpoint('best_model.pt')
                    elif self.config.train.early_stopping_patience is not None:
                        stale = epoch - self.best_epoch
                        if stale >= self.config.train.early_stopping_patience:
                            logger.info(
                                f"Early stopping at epoch {epoch+1}: "
                                f"no val acc improvement > {min_delta:.4f} for {stale} epochs"
                            )
                            break

                self.scheduler.step()

                # Periodic auto-save (every 5 epochs)
                if (epoch + 1) % 5 == 0:
                    self.save_checkpoint(f'auto_epoch_{epoch+1}.pt')
                    self.save_checkpoint('latest.pt')

        except KeyboardInterrupt:
            logger.info(f"\n⏸️  Training interrupted at epoch {self.current_epoch+1}/{epochs}")
            self.save_checkpoint('interrupted.pt')
            logger.info(f"💾 Saved to interrupted.pt — resume with: --resume interrupted.pt")

        # Final save
        self.save_checkpoint('latest.pt')
        logger.info(f"✅ Training done. Best val acc: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}")
        return self.history

    def resume(self, checkpoint_path: str, train_loader, val_loader,
               num_epochs: Optional[int] = None):
        """Resume training from a checkpoint."""
        ckpt = self.load_checkpoint(checkpoint_path)
        self.start_epoch = ckpt.get('epoch', 0) + 1
        logger.info(f"🔄 Resuming from epoch {self.start_epoch+1}")
        logger.info(f"   Best val acc so far: {self.best_val_acc:.4f} (epoch {self.best_epoch+1})")
        return self.train(train_loader, val_loader, num_epochs)

    # ─── Checkpoint I/O ──────────────────────────────────────────

    def save_checkpoint(self, filename: str):
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'history': self.history,
            'config': self.config,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
        }
        torch.save(ckpt, output_dir / filename)
        logger.info(f"💾 Checkpoint: {output_dir / filename}")

    def load_checkpoint(self, path: str) -> dict:
        """Load checkpoint. Returns the checkpoint dict for resume."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if ckpt.get('scaler_state_dict') and self.scaler:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.best_val_acc = ckpt.get('best_val_acc', 0.0)
        self.best_epoch = ckpt.get('best_epoch', 0)
        self.history = ckpt.get('history', {})
        self.current_epoch = ckpt.get('epoch', 0)
        logger.info(f"📂 Loaded checkpoint from {path} (epoch {self.current_epoch+1})")
        return ckpt

    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint to resume from."""
        output_dir = Path(self.config.output_dir)
        if not output_dir.exists():
            return None
        # Priority: interrupted > latest > auto_epoch_*
        for name in ['interrupted.pt', 'latest.pt']:
            p = output_dir / name
            if p.exists():
                return str(p)
        # Fallback: most recent auto_epoch checkpoint
        autos = sorted(output_dir.glob('auto_epoch_*.pt'), key=lambda x: x.stat().st_mtime, reverse=True)
        return str(autos[0]) if autos else None

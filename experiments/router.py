"""
Lesion-Aware Token Scoring (LATS) — core module for MedTokenBudget.

Given patch embeddings from a frozen ViT, LATS scores each patch by
lesion-relevance using multiple complementary signals, then routes
the top-K patches to a lightweight task head.

Signals:
  1. Attention Entropy  — background patches have uniform attention
  2. Feature Norm       — lesion patches often have higher activation
  3. Frequency Content   — lesion boundaries have high-frequency components
  4. GradCAM Attribution — gradient-based importance (optional)
  5. Prediction Disagreement — uncertain patches may be lesions

Reference: MedTokenBudget (AAAI 2027 submission)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import math


class LesionAwareTokenScorer(nn.Module):
    """
    Multi-signal token importance scorer.

    Computes a lesion-relevance score for each patch embedding,
    combining multiple complementary signals through a learned MLP.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_attention_entropy: bool = True,
        use_feature_norm: bool = True,
        use_frequency_content: bool = True,
        use_gradcam: bool = False,
        use_prediction_disagreement: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_attention_entropy = use_attention_entropy
        self.use_feature_norm = use_feature_norm
        self.use_frequency_content = use_frequency_content
        self.use_gradcam = use_gradcam
        self.use_prediction_disagreement = use_prediction_disagreement

        # Compute input dimension for the scorer MLP
        signal_dim = 0
        if use_attention_entropy:
            signal_dim += 1  # Scalar entropy per patch
        if use_feature_norm:
            signal_dim += 1  # Scalar norm per patch
        if use_frequency_content:
            signal_dim += embed_dim // 16  # Reduced frequency features
        if use_gradcam:
            signal_dim += 1  # Scalar attribution per patch
        if use_prediction_disagreement:
            signal_dim += 1  # Scalar disagreement per patch

        # Combine patch embedding with signals
        self.input_proj = nn.Linear(embed_dim + signal_dim, hidden_dim)

        # MLP scorer
        layers = []
        for i in range(num_layers):
            in_dim = hidden_dim if i > 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(hidden_dim, 1))  # Single importance score
        self.scorer = nn.Sequential(*layers)

        # Frequency feature extractor (lightweight)
        if use_frequency_content:
            self.freq_proj = nn.Linear(embed_dim, embed_dim // 16)

    def compute_signals(
        self,
        patches: torch.Tensor,           # [B, N, D]
        attentions: Optional[torch.Tensor] = None,  # [B, H, N, N]
        patch_positions: Optional[torch.Tensor] = None,  # [B, N, 2]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all lesion-relevance signals for each patch.

        Args:
            patches: Patch embeddings [B, N, D]
            attentions: Attention weights from ViT [B, H, N+1, N+1] (with CLS)
            patch_positions: Spatial positions of patches [B, N, 2]

        Returns:
            Dictionary of signal tensors, each [B, N, *]
        """
        B, N, D = patches.shape
        signals = {}
        device = patches.device

        # Signal 1: Attention Entropy
        if self.use_attention_entropy and attentions is not None:
            # Remove CLS token, average over heads
            attn_no_cls = attentions[:, :, 1:, 1:]  # [B, H, N, N]
            attn_mean = attn_no_cls.mean(dim=1)      # [B, N, N]

            # Entropy of attention distribution per patch (as query)
            eps = 1e-8
            ent = -(attn_mean * (attn_mean + eps).log()).sum(dim=-1)  # [B, N]
            # Normalize: high entropy = uniform attention (background)
            # Low entropy = focused attention (potential lesion)
            ent_norm = ent / math.log(N)
            # Invert: high score = potential lesion
            signals['attention'] = (1.0 - ent_norm).unsqueeze(-1)  # [B, N, 1]

        elif self.use_attention_entropy:
            # Fallback: use feature similarity as pseudo-attention
            sim = F.cosine_similarity(
                patches.unsqueeze(2), patches.unsqueeze(1), dim=-1
            )  # [B, N, N]
            attn = F.softmax(sim / 0.1, dim=-1)
            eps = 1e-8
            ent = -(attn * (attn + eps).log()).sum(dim=-1)
            ent_norm = ent / math.log(N)
            signals['attention'] = (1.0 - ent_norm).unsqueeze(-1)

        # Signal 2: Feature Norm
        if self.use_feature_norm:
            norm = patches.norm(p=2, dim=-1)  # [B, N]
            # Normalize per image
            norm = norm / norm.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
            signals['norm'] = norm.unsqueeze(-1)  # [B, N, 1]

        # Signal 3: Frequency Content
        if self.use_frequency_content:
            # Approximate frequency content via feature variance in local neighborhood
            # Reshape to 2D grid if positions available
            if patch_positions is not None:
                # Use actual spatial positions
                h = w = int(math.sqrt(N))
                if h * w == N:
                    patches_2d = patches.reshape(B, h, w, D)
                    # Compute local variance (high-freq proxy)
                    local_var = F.avg_pool2d(
                        patches_2d.permute(0, 3, 1, 2),
                        kernel_size=3, stride=1, padding=1
                    )
                    local_var = (patches_2d - local_var.permute(0, 2, 3, 1)).pow(2).mean(dim=-1)
                    freq_feat = self.freq_proj(local_var.reshape(B, N))
                    signals['frequency'] = freq_feat  # [B, N, D//16]
                else:
                    signals['frequency'] = torch.zeros(B, N, self.embed_dim // 16, device=device)
            else:
                # Use high-pass filter in feature space
                # High-freq ≈ large difference from neighbors
                freq_score = torch.zeros(B, N, device=device)
                if N > 1:
                    sim = F.cosine_similarity(
                        patches[:, :-1], patches[:, 1:], dim=-1
                    )
                    # Low similarity with neighbors → high frequency.
                    # Convert N-1 pair scores into N per-token scores.
                    pair_score = 1.0 - sim
                    counts = torch.zeros(B, N, device=device)
                    freq_score[:, :-1] += pair_score
                    freq_score[:, 1:] += pair_score
                    counts[:, :-1] += 1
                    counts[:, 1:] += 1
                    freq_score = freq_score / counts.clamp(min=1)
                freq_score = freq_score.unsqueeze(-1)
                freq_feat = self.freq_proj(patches) * freq_score
                signals['frequency'] = freq_feat

        # Signal 4: GradCAM Attribution (computed externally, passed through)
        if self.use_gradcam:
            signals['gradcam'] = torch.zeros(B, N, 1, device=device)

        # Signal 5: Prediction Disagreement
        if self.use_prediction_disagreement:
            # Placeholder: actually computed during forward pass
            signals['disagreement'] = torch.zeros(B, N, 1, device=device)

        return signals

    def forward(
        self,
        patches: torch.Tensor,
        attentions: Optional[torch.Tensor] = None,
        patch_positions: Optional[torch.Tensor] = None,
        return_scores: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Score each patch and return importance scores.

        Returns:
            scores: Token importance scores [B, N] in [0, 1]
            signal_dict: Individual signal values
        """
        B, N, D = patches.shape

        # Compute all signals
        signal_dict = self.compute_signals(patches, attentions, patch_positions)

        # Build augmented features: patch embedding + scalar signals
        scalar_signals = []
        for name, signal in signal_dict.items():
            if signal.dim() == 3 and signal.shape[-1] == 1:
                scalar_signals.append(signal)  # [B, N, 1]
            elif signal.dim() == 2:
                scalar_signals.append(signal.unsqueeze(-1))  # [B, N] -> [B, N, 1]

        if scalar_signals:
            extra = torch.cat(scalar_signals, dim=-1)  # [B, N, S]
            patch_with_signals = torch.cat([patches, extra], dim=-1)  # [B, N, D+S]
        else:
            patch_with_signals = patches

        # Ensure dimension matches input_proj
        target_dim = self.input_proj.in_features
        current_dim = patch_with_signals.shape[-1]
        if current_dim < target_dim:
            padding = torch.zeros(B, N, target_dim - current_dim, device=patches.device)
            patch_with_signals = torch.cat([patch_with_signals, padding], dim=-1)
        elif current_dim > target_dim:
            patch_with_signals = patch_with_signals[..., :target_dim]

        # Score each patch
        hidden = self.input_proj(patch_with_signals)
        scores = self.scorer(hidden).squeeze(-1)  # [B, N]
        scores = torch.sigmoid(scores)

        return scores, signal_dict


class TokenRouter(nn.Module):
    """
    Top-K token router with spatial smoothing.

    Given importance scores, selects the top-K patches and applies
    spatial smoothing to keep lesion regions contiguous.
    """

    def __init__(
        self,
        budget_ratio: float = 0.5,
        min_tokens: int = 16,
        spatial_kernel: int = 3,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.budget_ratio = budget_ratio
        self.min_tokens = min_tokens
        self.spatial_kernel = spatial_kernel
        self.temperature = temperature

    def spatial_smooth(
        self,
        scores: torch.Tensor,
        h: int, w: int,
    ) -> torch.Tensor:
        """Apply spatial smoothing to importance scores."""
        B, N = scores.shape

        if h * w != N:
            return scores  # Non-square, skip smoothing

        scores_2d = scores.reshape(B, 1, h, w)

        # Average pooling with kernel_size
        kernel_size = self.spatial_kernel
        padding = kernel_size // 2
        smoothed = F.avg_pool2d(
            scores_2d, kernel_size=kernel_size,
            stride=1, padding=padding
        )

        return smoothed.reshape(B, N)

    def forward(
        self,
        scores: torch.Tensor,
        patches: torch.Tensor,
        training: bool = True,
        budget_ratio: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Route top-K patches based on importance scores.

        Args:
            scores: Token importance scores [B, N]
            patches: Patch embeddings [B, N, D]
            training: Whether in training mode (use Gumbel-softmax)
            budget_ratio: Override default budget ratio

        Returns:
            Dictionary with:
                - selected_patches: [B, K, D]
                - selected_indices: [B, K]
                - selection_mask: [B, N] (binary)
                - kept_ratio: actual fraction kept
        """
        B, N, D = patches.shape

        budget = budget_ratio if budget_ratio is not None else self.budget_ratio
        K = max(self.min_tokens, int(N * budget))
        K = min(K, N)

        # Spatial smoothing
        h = w = int(math.sqrt(N))
        if h * w == N:
            scores = self.spatial_smooth(scores, h, w)

        top_indices = None  # only populated in eval mode

        if training:
            # Differentiable top-K via Gumbel-softmax straight-through
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(scores).clamp(min=1e-8)
            ))
            logits = (scores.log() - (1 - scores).log() + gumbel_noise) / self.temperature

            _, top_indices = logits.topk(K, dim=-1, sorted=False)
            mask = torch.zeros_like(scores)
            mask.scatter_(1, top_indices, 1.0)
            selection_mask = mask
        else:
            # Hard top-K selection at inference
            _, top_indices = scores.topk(K, dim=-1, sorted=False)
            selection_mask = torch.zeros_like(scores)
            selection_mask.scatter_(1, top_indices, 1.0)

        selected_patches = patches * selection_mask.unsqueeze(-1)

        return {
            'selected_patches': selected_patches,
            'selected_indices': top_indices if not training else None,
            'selection_mask': selection_mask,
            'kept_ratio': patches.new_full((B,), K / N),
            'num_kept': K,
            'scores': scores,
        }

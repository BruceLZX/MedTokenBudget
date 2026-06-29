"""
MedTokenBudget model: Frozen ViT backbone + LATS router + lightweight task head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math

from router import LesionAwareTokenScorer, TokenRouter


class MedTokenBudget(nn.Module):
    """
    Lesion-preserving token-routed medical vision model.

    Pipeline:
        Image → Frozen ViT → Patch Embeddings
                → LATS Scorer → Token Importance Scores
                → TokenRouter → Top-K Patches
                → Lightweight Head → Prediction
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int = 768,
        num_classes: int = 8,
        num_patches: int = 196,
        router_config: Optional[Dict] = None,
        head_config: Optional[Dict] = None,
    ):
        super().__init__()

        # Default configs
        if router_config is None:
            router_config = {}
        if head_config is None:
            head_config = {}

        # Frozen backbone
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.embed_dim = embed_dim
        self.num_patches = num_patches

        # LATS scorer
        self.scorer = LesionAwareTokenScorer(
            embed_dim=embed_dim,
            hidden_dim=router_config.get('hidden_dim', 128),
            num_layers=router_config.get('num_layers', 3),
            dropout=router_config.get('dropout', 0.1),
            use_attention_entropy=router_config.get('use_attention_entropy', True),
            use_feature_norm=router_config.get('use_feature_norm', True),
            use_frequency_content=router_config.get('use_frequency_content', True),
            use_gradcam=router_config.get('use_gradcam', False),
            use_prediction_disagreement=router_config.get('use_prediction_disagreement', False),
        )

        # Token router
        self.router = TokenRouter(
            budget_ratio=router_config.get('token_budget_ratio', 0.5),
            min_tokens=router_config.get('min_tokens', 16),
            spatial_kernel=router_config.get('spatial_smoothing_kernel', 3),
            temperature=router_config.get('temperature', 1.0),
        )

        # Lightweight task head
        head_type = head_config.get('type', 'mlp')
        head_hidden = head_config.get('hidden_dim', 256)
        if head_type == 'linear':
            self.head = nn.Linear(embed_dim, num_classes)
        elif head_type == 'transformer':
            # Lightweight 2-layer transformer on selected patches
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=8,
                dim_feedforward=head_hidden,
                dropout=0.1, batch_first=True,
            )
            self.head = nn.Sequential(
                nn.TransformerEncoder(encoder_layer, num_layers=2),
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, num_classes),
            )
        else:  # mlp (default)
            self.head = nn.Sequential(
                nn.Linear(embed_dim, head_hidden),
                nn.LayerNorm(head_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(head_hidden, head_hidden // 2),
                nn.LayerNorm(head_hidden // 2),
                nn.GELU(),
                nn.Linear(head_hidden // 2, num_classes),
            )

        # Budget for current forward pass (can be overridden)
        self.current_budget = router_config.get('token_budget_ratio', 0.5)

    def _attention_from_block(self, block: nn.Module, tokens: torch.Tensor) -> Optional[torch.Tensor]:
        """Return last-block attention weights when the backbone exposes qkv."""
        attn_mod = getattr(block, "attn", None)
        qkv_layer = getattr(attn_mod, "qkv", None)
        if attn_mod is None or qkv_layer is None:
            return None

        B, N, C = tokens.shape
        num_heads = getattr(attn_mod, "num_heads", None)
        if num_heads is None:
            return None
        head_dim = C // num_heads
        scale = getattr(attn_mod, "scale", head_dim ** -0.5)
        qkv = qkv_layer(tokens).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        return (q @ k.transpose(-2, -1) * scale).softmax(dim=-1)

    def extract_patches(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract patch embeddings from frozen ViT backbone.
        Returns patches [B, N, D] and attention weights (or None).
        """
        attentions = None
        with torch.no_grad():
            # DINOv2 (loaded via torch.hub)
            if hasattr(self.backbone, 'prepare_tokens_with_masks') and hasattr(self.backbone, 'blocks'):
                tokens = self.backbone.prepare_tokens_with_masks(x, None)
                for blk in self.backbone.blocks[:-1]:
                    tokens = blk(tokens)
                attentions = self._attention_from_block(self.backbone.blocks[-1], tokens)
                tokens = self.backbone.blocks[-1](tokens)
                if hasattr(self.backbone, 'norm'):
                    tokens = self.backbone.norm(tokens)
                patches = tokens[:, 1:, :]

            elif hasattr(self.backbone, 'forward_features'):
                out = self.backbone.forward_features(x)
                if isinstance(out, dict):
                    patches = out.get('x_norm_patchtokens')
                    if patches is None:
                        patches = out.get('x_prenorm', list(out.values())[0])
                elif isinstance(out, (list, tuple)):
                    patches = out[0]
                else:
                    patches = out
                if patches.dim() == 3 and patches.shape[1] == self.num_patches + 1:
                    patches = patches[:, 1:, :]

            # timm ViT or similar
            elif hasattr(self.backbone, 'blocks'):
                h = self.backbone.patch_embed(x)
                if self.backbone.cls_token is not None:
                    cls_tok = self.backbone.cls_token.expand(h.shape[0], -1, -1)
                    h = torch.cat((cls_tok, h), dim=1)
                if hasattr(self.backbone, 'pos_embed'):
                    h = h + self.backbone.pos_embed
                if hasattr(self.backbone, 'pos_drop'):
                    h = self.backbone.pos_drop(h)
                for blk in self.backbone.blocks[:-1]:
                    h = blk(h)
                attentions = self._attention_from_block(self.backbone.blocks[-1], h)
                h = self.backbone.blocks[-1](h)
                if hasattr(self.backbone, 'norm'):
                    h = self.backbone.norm(h)
                patches = h[:, 1:, :]

            # Fallback: try calling backbone directly and strip CLS
            else:
                out = self.backbone(x)
                if isinstance(out, dict):
                    patches = out.get('last_hidden_state', list(out.values())[0])
                elif isinstance(out, (list, tuple)):
                    patches = out[0]
                else:
                    patches = out
                if patches.dim() == 3 and patches.shape[1] == self.num_patches + 1:
                    patches = patches[:, 1:, :]

            # Ensure correct shape
            if patches.dim() == 4:  # [B, H, W, D]
                patches = patches.flatten(1, 2)
            B, N, D = patches.shape

            # Align to expected num_patches
            if N < self.num_patches:
                padding = torch.zeros(B, self.num_patches - N, D, device=patches.device)
                patches = torch.cat([patches, padding], dim=1)
            elif N > self.num_patches:
                patches = patches[:, :self.num_patches, :]
                if attentions is not None:
                    keep = self.num_patches + 1
                    attentions = attentions[:, :, :keep, :keep]

        return patches, attentions

    def forward(
        self,
        x: torch.Tensor,
        budget_ratio: Optional[float] = None,
        return_routing_info: bool = False,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with token routing.

        Args:
            x: Input images [B, 3, H, W]
            budget_ratio: Token budget override
            return_routing_info: Whether to return routing details
            return_all: Whether to return all outputs (for analysis)

        Returns:
            Dictionary with logits and optional routing information
        """
        B = x.shape[0]

        # Extract patch embeddings
        patches, attentions = self.extract_patches(x)

        # Score patches
        scores, signal_dict = self.scorer(patches, attentions)

        # Route tokens
        budget = budget_ratio if budget_ratio is not None else self.current_budget
        route_result = self.router(
            scores, patches,
            training=self.training,
            budget_ratio=budget,
        )

        # Task head on selected patches
        selected = route_result['selected_patches']  # [B, N, D] with zeros for dropped
        mask = route_result['selection_mask'].unsqueeze(-1)  # [B, N, 1]

        # Option 1: Pool selected patches (mean pooling of kept patches)
        kept_count = mask.sum(dim=1).clamp(min=1)  # [B, 1, 1]
        pooled = selected.sum(dim=1) / kept_count  # [B, D]

        logits = self.head(pooled)

        output = {'logits': logits}

        if return_routing_info:
            output.update({
                'scores': scores,
                'patches': patches,  # needed for attention distillation loss
                'attentions': attentions,
                'selection_mask': route_result['selection_mask'],
                'kept_ratio': route_result['kept_ratio'],
                'num_kept': route_result['num_kept'],
                'signal_dict': signal_dict,
            })

        if return_all:
            output.update({
                'patches': patches,
                'selected_patches': selected,
                'pooled': pooled,
            })

        return output

    def set_budget(self, ratio: float):
        """Update token budget for inference."""
        self.current_budget = ratio

    def get_trainable_params(self):
        """Return only trainable parameters (scorer + head)."""
        return list(self.scorer.parameters()) + list(self.head.parameters())

    def compute_lesion_retention(
        self,
        selection_mask: torch.Tensor,
        lesion_mask: torch.Tensor,
    ) -> float:
        """
        Compute fraction of lesion patches retained by the router.

        Args:
            selection_mask: Binary mask from router [B, N]
            lesion_mask: Ground-truth lesion mask [B, N] (1 = lesion)

        Returns:
            Lesion retention rate
        """
        lesion_patches = lesion_mask.sum()
        if lesion_patches == 0:
            return 1.0  # No lesion → perfect retention (vacuously)

        retained_lesions = (selection_mask * lesion_mask).sum()
        return (retained_lesions / lesion_patches).item()

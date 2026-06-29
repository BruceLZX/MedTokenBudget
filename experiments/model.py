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
        num_patches: int = 196,  # 224/16 = 14, 14^2 = 196
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

        # CLS token handling
        self.use_cls_token = True
        self.cls_head = nn.Linear(embed_dim, num_classes)  # For no-pruning baseline

        # Budget for current forward pass (can be overridden)
        self.current_budget = router_config.get('token_budget_ratio', 0.5)

    def extract_patches(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract patch embeddings from frozen ViT backbone.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            patches: Patch embeddings [B, N, D]
            attentions: Attention weights (if available) or None
        """
        with torch.no_grad():
            # Different backbones have different interfaces
            # We need the intermediate patch embeddings before the final head

            if hasattr(self.backbone, 'get_intermediate_layers'):
                # DINOv2
                outputs = self.backbone.get_intermediate_layers(
                    x, n=1, return_class_token=True
                )
                patches = outputs[0][0]  # Patch tokens without CLS
                # Shape: [B, N+1, D] → remove CLS → [B, N, D]
                if patches.shape[1] == self.num_patches + 1:
                    patches = patches[:, 1:, :]

            elif hasattr(self.backbone, 'blocks'):
                # timm ViT: forward through blocks manually
                x = self.backbone.patch_embed(x)
                if self.backbone.cls_token is not None:
                    cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
                    x = torch.cat((cls_token, x), dim=1)
                x = x + self.backbone.pos_embed

                attentions = []
                for block in self.backbone.blocks:
                    x = block(x)
                    if hasattr(block, 'attn') and hasattr(block.attn, 'attn_weights'):
                        attentions.append(block.attn.attn_weights)

                patches = x[:, 1:, :]  # Remove CLS
                # Get last layer attention
                attn = attentions[-1] if attentions else None

            elif hasattr(self.backbone, 'encoder'):
                # SAM-style encoder
                x = self.backbone.patch_embed(x)
                if self.backbone.pos_embed is not None:
                    x = x + self.backbone.pos_embed
                patches = self.backbone.encoder(x)
                attn = None

            else:
                # Generic: try forward_features
                try:
                    patches = self.backbone.forward_features(x)
                    if isinstance(patches, (list, tuple)):
                        patches = patches[0]
                    if patches.dim() == 3 and patches.shape[1] == self.num_patches + 1:
                        patches = patches[:, 1:, :]
                except Exception:
                    # Last resort: use the full model and extract manually
                    # This is a fallback; user should implement proper extraction
                    raise NotImplementedError(
                        "Backbone interface not supported. "
                        "Implement extract_patches() for your backbone."
                    )
                attn = None

            # Ensure correct shape
            if patches.dim() == 4:  # [B, H, W, D] → [B, N, D]
                patches = patches.flatten(1, 2)
            B, N, D = patches.shape

            # Pad or truncate to expected num_patches
            if N < self.num_patches:
                padding = torch.zeros(B, self.num_patches - N, D, device=patches.device)
                patches = torch.cat([patches, padding], dim=1)
            elif N > self.num_patches:
                patches = patches[:, :self.num_patches, :]

        return patches, None  # attn is None for now (extracting attention is model-specific)

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
        pooled = (selected * mask).sum(dim=1) / kept_count  # [B, D]

        logits = self.head(pooled)

        # CLS baseline logits (no pruning)
        cls_logits = None
        if self.use_cls_token and not self.training:
            with torch.no_grad():
                cls_patches, _ = self.extract_patches(x)
                # Use all patches pooled
                cls_pooled = cls_patches.mean(dim=1)
                cls_logits = self.cls_head(cls_pooled)

        output = {'logits': logits}

        if return_routing_info:
            output.update({
                'scores': scores,
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
                'cls_logits': cls_logits,
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

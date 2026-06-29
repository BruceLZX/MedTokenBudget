"""
HuggingFace Spaces Demo: MedTokenBudget — Lesion-Preserving Token Routing.

Interactive Gradio app for visualizing:
  1. Token importance heatmap overlay on medical images
  2. Lesion retention vs token budget tradeoff
  3. Comparison with baselines (random, attention-based)
"""

import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import MedTokenBudget
from router import LesionAwareTokenScorer


def load_example():
    """Generate demo visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Simulated ISIC skin lesion image
    np.random.seed(42)
    img = np.random.rand(224, 224, 3) * 0.3
    # Add a "lesion" (darker irregular region)
    xx, yy = np.meshgrid(np.arange(224), np.arange(224))
    lesion_mask = ((xx - 120)**2 / 30**2 + (yy - 100)**2 / 20**2) < 1
    img[lesion_mask] *= 0.3
    img[lesion_mask, 0] *= 1.5  # Reddish lesion

    axes[0].imshow(img)
    axes[0].set_title("Original Image (simulated skin lesion)")
    axes[0].axis('off')

    # Simulated token importance heatmap
    heatmap = np.zeros((14, 14))
    lesion_region = ((np.arange(14)[:, None] - 7)**2 / 2**2 + (np.arange(14)[None, :] - 6)**2 / 1.5**2) < 1
    heatmap[lesion_region] = np.random.uniform(0.6, 1.0, lesion_region.sum())
    heatmap[~lesion_region] = np.random.uniform(0.0, 0.3, (~lesion_region).sum())

    axes[1].imshow(img, alpha=0.6)
    im = axes[1].imshow(heatmap, cmap='hot', alpha=0.5, interpolation='nearest')
    axes[1].set_title("Token Importance Scores (LATS)")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], label='Importance')

    plt.tight_layout()

    return fig, """
    ## MedTokenBudget Demo

    This is a conceptual demo. For full results, run the experiment pipeline.

    **How it works:**
    1. Frozen DINOv2 ViT extracts 196 patches from the image
    2. LATS (Lesion-Aware Token Scoring) rates each patch by lesion relevance
    3. Top-K patches are routed to a lightweight classifier
    4. Lesion patches are preserved even under aggressive budgets

    **Key Metrics:**
    - Lesion Retention Rate @ K tokens
    - Classification Accuracy vs Token Budget
    """


with gr.Blocks(title="MedTokenBudget Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏥 MedTokenBudget: Lesion-Preserving Token Routing

    **AAAI 2027 Submission** | *"Not all patches are equal — why waste compute on healthy tissue?"*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Concept Visualization")
            run_btn = gr.Button("Generate Demo", variant="primary", size="lg")

        with gr.Column(scale=2):
            plot_output = gr.Plot(label="Token Importance Visualization")

    info_output = gr.Markdown()

    run_btn.click(load_example, outputs=[plot_output, info_output])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

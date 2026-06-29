# Paper Outline: MedTokenBudget — Lesion-Preserving Token Routing for Medical Vision Models

**Target:** AAAI 2027
**Status:** 🔬 Research Phase

---

## Title

**"MedTokenBudget: Lesion-Preserving Token Routing for Efficient Medical Vision Models"**

Alternative: *"Not All Patches Are Equal: Lesion-Aware Token Budgeting for Medical Image Understanding"*

---

## 1. Introduction

### Problem
Medical images are dominated by normal tissue. A chest X-ray is mostly black lung fields; a skin image is mostly healthy skin; a brain MRI is mostly normal parenchyma. Yet vision transformers process ALL patches equally, wasting compute on uninformative regions while potentially losing small lesions.

### Gap
- Token pruning exists (DynamicViT, ToMe, EViT) but is **task-agnostic** — it drops tokens based on attention scores, not medical relevance
- No prior work studies **lesion-preserving token routing** for medical imaging
- Current pruning methods may systematically drop small pathological regions

### Contribution
1. **Lesion-Aware Token Scoring (LATS)**: A lightweight module that scores patch importance based on multiple signals (attention entropy, feature norm, frequency content, prediction uncertainty)
2. **Budget-Constrained Token Routing**: Dynamic allocation of token budget across image regions, preserving high-scoring (potentially lesional) patches
3. **Systematic evaluation** across 3 modalities (dermoscopy, brain MRI, pathology) showing LATS preserves small-lesion accuracy under aggressive token budgets
4. **Analysis of failure modes**: When do standard pruning methods drop lesions?

---

## 2. Related Work

### Token Pruning (General Vision)
- DynamicViT (Rao et al., 2021): Hierarchical token sparsification
- ToMe (Bolya et al., 2023): Token merging via bipartite matching
- EViT (Liang et al., 2022): Attentive token selection
- **All task-agnostic — no medical awareness**

### Medical Image Efficiency
- MedSpaformer (AAAI 2026): Token sparsification for medical time series (not images)
- No prior medical token routing paper

### Lesion Detection / Weakly-Supervised Localization
- CAM, GradCAM, Ablation-CAM
- We use these as **inspiration for scoring**, not as methods

---

## 3. Method: Lesion-Aware Token Scoring (LATS)

### 3.1 Overview
```
Image → Frozen ViT → Patch Embeddings [N, D]
                         ↓
                  LATS Scoring Module
                         ↓
              Token Importance Scores [N]
                         ↓
           Top-K Token Selection (Budget B)
                         ↓
        Lightweight Classifier / Segmenter Head
```

### 3.2 Scoring Signals
| Signal | Intuition | Computation |
|--------|-----------|-------------|
| **Attention Entropy** | Background patches attend uniformly; lesion patches have focused attention | H = -Σ α log α over attention heads |
| **Feature Norm** | Lesion features often have higher activation magnitude | ||z_i||₂ |
| **Frequency Content** | Lesion boundaries have high-frequency components | FFT magnitude of patch |
| **GradCAM Attribution** | Gradient-based importance (requires one backward pass) | ∂ŷ/∂A · A |
| **Prediction Disagreement** | Patches where model is uncertain | Entropy of patch-level predictions |

### 3.3 Budget Allocation
- Global budget B: total tokens to keep (e.g., 25%, 50%, 75% of all patches)
- Scores are normalized and thresholded
- **Spatial smoothing**: Nearby high-score patches are kept together (avoid isolated token retention)

### 3.4 Training
- LATS is a lightweight MLP (3 layers, ~50K params)
- Trained with: classification loss + token budget regularization + lesion localization auxiliary loss
- Frozen ViT backbone (DINOv2, MedMAE, or SAM encoder)

---

## 4. Experiments

### 4.1 Datasets
| Dataset | Modality | Task | Key Property |
|---------|----------|------|-------------|
| ISIC 2019 | Dermoscopy | 8-class skin lesion | Small lesions in large images |
| BRISC | Brain MRI | 4-class tumor | Tumors vary in size/location |
| PathMNIST | Colon pathology | 9-class tissue | Cellular-level features |

### 4.2 Baselines
- No pruning (upper bound)
- Random token dropping
- DynamicViT pruning
- ToMe merging
- EViT selection
- **LATS (ours)**

### 4.3 Metrics
- Classification accuracy vs. token budget
- **Lesion Recall@K**: fraction of lesion-containing patches retained in top-K
- F1 on small-lesion subset
- FLOPs / inference time

### 4.4 Key Ablations
- Individual scoring signal contribution
- Spatial smoothing effect
- Cross-dataset generalization of LATS
- Backbone robustness (DINOv2 vs MedMAE vs SAM)

---

## 5. Expected Key Figures

1. **Accuracy vs. Token Budget** — LATS Pareto-dominates baselines, especially at low budgets
2. **Lesion Retention Rate** — LATS retains >90% of lesion patches even at 25% budget
3. **Qualitative Visualization** — Heatmap overlay of kept/dropped tokens on ISIC lesions
4. **Failure Case Analysis** — Examples where LATS fails (diffuse diseases, multi-focal lesions)

---

## 6. Timeline (8 weeks)

| Week | Tasks |
|------|-------|
| 1 | Implement LATS module + MedMNIST data pipeline |
| 2 | ISIC + BRISC data loading, baseline implementations |
| 3 | Full experiment run on ISIC + MedMNIST |
| 4 | BRISC experiments, ablation studies |
| 5 | Lesion retention analysis + visualization |
| 6 | Write paper (Sections 1-4) |
| 7 | Write paper (Sections 5-7), polish figures |
| 8 | Final polish, supplementary, code release |

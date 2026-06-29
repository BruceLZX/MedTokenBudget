# MedTokenBudget: Lesion-Preserving Token Routing

**Target:** AAAI 2027 | **Status:** 🔬 In Progress

> *"Not all patches are equal. Medical images are mostly normal tissue — why waste compute on background patches when a tiny lesion holds the diagnosis?"*

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Quick sanity check (10 epochs, PathMNIST, ~5 min)
python run_experiments.py --mode quick

# 3. Full ISIC training (50 epochs, ~2 hours on A100)
python run_experiments.py --mode full --dataset isic

# 4. Budget sweep after training (evaluates all budgets + baselines)
python run_experiments.py --mode sweep --dataset isic
```

## How to Run (Step by Step)

### Step 1: MedMNIST Sanity Check (CPU, 5 min)

```bash
pip install medmnist torch torchvision timm tqdm scikit-learn

# Quick test — verifies the pipeline works
python run_experiments.py --mode quick --device cpu
```

**Expected output:**
```
Loading medmnist dataset...
Train: 89996, Val: 10004, Classes: 9
Loading dino_v2 backbone...
Model: 86,000,000 total params, 1,200,000 trainable
Epoch 1 (budget=1.00): 100% | loss: 1.8234 | acc: 0.452
...
Training complete. Best val acc: 0.XXXX at epoch X
```

### Step 2: ISIC 2019 Training (A100, ~2 hours)

```bash
# Download ISIC 2019 first:
# https://challenge.isic-archive.com/data/#2019
# Or use auto-fallback to DermaMNIST (built into datasets.py)

python run_experiments.py --mode full --dataset isic --device cuda
```

### Step 3: Budget Sweep (A100, ~30 min)

```bash
python run_experiments.py --mode sweep --dataset isic --device cuda
```

**Evaluates LATS + 6 baselines at budgets [10%, 25%, 50%, 75%, 100%]**

Output: `results/med_token_budget/budget_sweep.json`

### Step 4: BRISC Brain MRI

```bash
# Download from Kaggle: https://www.kaggle.com/datasets/briscdataset/brisc2025/
python run_experiments.py --mode full --dataset brisc
```

### Step 5: All Experiments

```bash
python run_experiments.py --mode all --device cuda
```

---

## Method: LATS (Lesion-Aware Token Scoring)

```
Image → Frozen ViT → Patch Embeddings [196, 768]
                            ↓
                     LATS Scoring Module
                     ┌─────────────────────┐
                     │ Attention Entropy    │ ← Uniform attn = background
                     │ Feature Norm         │ ← High norm = lesion
                     │ Frequency Content    │ ← High freq = boundaries
                     └─────────────────────┘
                            ↓
                  Token Importance Scores [196]
                            ↓
               Top-K Token Selection (Budget B%)
                            ↓
            Lightweight MLP Classifier Head
```

## Datasets

| Dataset | Modality | Classes | Size | Auto-Download |
|---------|----------|---------|------|:---:|
| PathMNIST | Colon pathology | 9 | 107,180 | ✅ `pip install medmnist` |
| ISIC 2019 | Dermoscopy | 8 | 25,331 | ⚠️ Manual or DermaMNIST fallback |
| BRISC | Brain MRI | 4 | 6,000 | ⚠️ Kaggle/Figshare/Zenodo |

## Baselines (7 total)

| # | Baseline | Type | Training |
|---|----------|------|:---:|
| 1 | No pruning | Upper bound | — |
| 2 | Random | Pruning | ✗ |
| 3 | ToMe | Merging | ✗ |
| 4 | EViT | Pruning | ✗ |
| 5 | DynamicViT | Pruning | ✓ |
| 6 | Freq-Aware (Lee 2025) | Pruning | ✓ |
| 7 | **LATS (ours)** | Pruning | ✓ |

## Key Metrics

| Metric | Meaning | Target |
|--------|---------|--------|
| **Accuracy vs Budget** | Classification accuracy at each token budget | LATS Pareto-dominates |
| **Lesion Retention Rate** | % of lesion patches kept in Top-K | >90% at 25% budget |
| **Small-lesion F1** | F1 on samples with lesions <10% area | LATS > baselines |
| **Cross-modality transfer** | Train on ISIC → test on BRISC | Positive transfer |

## Hardware Requirements

| Mode | GPU | RAM | Time |
|------|-----|-----|------|
| Quick (MedMNIST) | CPU | 8GB | 5 min |
| Full (ISIC) | A100 | 32GB | 2 hrs |
| Full (BRISC) | A100 | 32GB | 1 hr |
| Budget sweep | A100 | 16GB | 30 min |
| All experiments | A100 | 32GB | 4-5 hrs |

## Project Structure

```
experiments/
├── config.py           # Configuration dataclasses
├── router.py           # LATS scorer + TokenRouter
├── model.py            # MedTokenBudget (backbone + LATS + head)
├── datasets.py         # MedMNIST, ISIC, BRISC loaders
├── train.py            # Trainer with budget curriculum
├── run_experiments.py  # Main entry point
├── app.py              # HuggingFace Spaces Gradio demo
├── requirements.txt    # pip dependencies
└── README.md           # This file
```

## Citation

```bibtex
@inproceedings{medtokenbudget2027,
  title={MedTokenBudget: Lesion-Preserving Token Routing for Efficient Medical Vision Models},
  author={...},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2027}
}
```

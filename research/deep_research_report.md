# Deep Research Report: MedTokenBudget — Competitor Map & Feasibility Analysis

**Date:** 2026-06-29
**Status:** ✅ Gap Confirmed — Specific Positioning Adjustments Required

---

## Executive Summary

**The core gap is genuine**: No paper does multi-signal lesion-aware token routing for standard medical image classification. However, the broad "token efficiency for medical images" space is heating up rapidly, with 6 papers in 2025-2026. We must position sharply.

**Key risk**: 3 WSI pathology token pruning papers + 1 frequency-aware token reduction paper are conceptually adjacent. We differentiate by: (1) standard image classification (not gigapixel WSI), (2) multi-signal lesion-awareness (not generic efficiency), (3) lesion retention as a primary metric.

---

## 1. COMPETITOR MAP

### 🔴 Tier 1: Direct Technical Overlap (Differentiate Carefully)

#### A. Prompt-based Dynamic Token Pruning (2025, arXiv:2506.16369)
- **What**: Box-prompt-guided spatial priors for token pruning in medical segmentation
- **Datasets**: ACDC, **ISIC** (same dataset!)
- **Overlap**: Uses ISIC, scores tokens by relevance, prunes for efficiency
- **Differentiation**: (a) They use external box prompts, we use self-supervised scoring. (b) Segmentation vs classification. (c) They don't study lesion retention or small-lesion preservation.

#### B. PRT: Patch Residual Transformer (2026, Scientific Reports)
- **What**: Top-K patch selection + linear stitch fusion for Alzheimer's MRI
- **Overlap**: Top-K patch selection for medical classification
- **Differentiation**: (a) Single-modality (brain MRI only). (b) Single scoring signal (learned ranking). (c) No small-lesion analysis. (d) Our multi-signal LATS is richer.

#### C. GAFormer (2026, Applied Soft Computing)
- **What**: Genetic algorithm for token selection in lung X-ray classification
- **Overlap**: Token importance evaluation for medical classification
- **Differentiation**: (a) GA is non-differentiable, LATS is end-to-end learnable. (b) Single-dataset. (c) No lesion-awareness.

#### D. Frequency-Aware Token Reduction (Lee et al., Nov 2025, arXiv:2511.21477)
- **What**: Decomposes attention into LF/HF components, preserves HF tokens, condenses LF tokens
- **Overlap**: Uses frequency information for token scoring. Preserves high-frequency details.
- **Results**: DeiT-S: 79.9% acc with 35% MACs reduction (outperforms ToMe, EViT, DynamicViT)
- **Differentiation**: (a) They decompose ATTENTION frequency, we analyze PATCH frequency content (different signal). (b) General vision, no medical adaptation. (c) No lesion-retention analysis. (d) We add attention entropy + feature norm + GradCAM as complementary signals.
- **Action**: MUST cite and run as baseline comparison. Show that LATS preserves lesions better than pure frequency-aware pruning.

### 🟡 Tier 2: Domain-Adjacent (Different Problem)

#### E. SLIM (NeurIPS 2025)
- **What**: Token pruning for WSI pathology vision-language modeling
- **Different**: Gigapixel WSI (thousands of patches per slide) vs standard images (196 patches)
- **Action**: Cite as related work in "token pruning for medical images", but clarify scale difference

#### F. SparseLearn (arXiv:2606.08641, Jun 2026)
- **What**: Learnable token sparsification for WSI, Soft Top-K, 0.78% token retention
- **Different**: WSI domain. Their "learnable scoring + Top-K" is technically similar to our approach.
- **Action**: CRITICAL — this is very recent (June 2026). Cite and differentiate: we operate on standard images (196 tokens → K tokens), they operate on WSIs (thousands → 32 tokens). Different scale, different challenges.

#### G. TC-SSA (arXiv:2603.01143, Mar 2026)
- **What**: Semantic slot aggregation for pathology WSI, 1.7% token compression
- **Different**: WSI domain, aggregation not pruning
- **Action**: Cite as related work

#### H. ViToS (ICML 2026)
- **What**: Dual-stream RL for medical VLM visual token pruning
- **Different**: VLM reasoning task, RL-based optimization
- **Action**: Cite as evidence that "medical token pruning is an emerging area"

#### I. TOP-RL (AAAI 2026)
- **What**: Task-optimized token pruning with RL for general VLMs
- **Different**: General VLM, not medical
- **Action**: Baseline citation for "RL-based token pruning exists but is heavy; our LATS is lightweight"

### 🟢 Tier 3: Complementary (Cite for Context)

#### J. MFTMamba-Unet (2026)
- **What**: Mamba+Transformer for small-size medical lesion segmentation
- **Cite for**: Evidence that small lesions are a recognized problem in medical imaging

#### K. EnFuseNet (2026)
- **What**: Long-tail skin lesion diagnosis with prototype enhancement
- **Cite for**: Long-tail problem in dermatology motivates lesion-preserving approaches

#### L. npj Digital Medicine (2025)
- **What**: Rare pathological lesion detection, up to 63.4% improvement for rare findings
- **Cite for**: Clinical importance of preserving rare/small findings

---

## 2. TOKEN PRUNING LANDSCAPE (General Vision)

### Baselines We Must Compare Against

| Method | Type | Training | Key Feature |
|--------|------|----------|-------------|
| **ToMe** | Merging | ✗ | Bipartite matching, training-free |
| **EViT** | Pruning | ✗ | Attention-based, plug-in |
| **DynamicViT** | Pruning | ✓ | Learned keep ratios |
| **DiffRate** | Hybrid | ✓ | Differentiable layer-wise ratios |
| **Freq-Aware (Lee 2025)** | Pruning | ✓ | HF token preservation, SOTA |
| **LATS (Ours)** | Pruning | ✓ | Multi-signal lesion-aware scoring |

### Key Insight from 2025 Survey
> "Token compression methods underperform on compact architectures without retraining — plug-in methods suffer ~50% accuracy drops." (Nguyen et al., Jul 2025)

This means LATS MUST be trained (not plug-and-play), which is fine since we're proposing a learned scorer.

---

## 3. LESION LOCALIZATION METHODS (Scoring Signal Sources)

| Method | Signal Type | Computational Cost | Use in LATS |
|--------|------------|-------------------|-------------|
| **GradCAM** | Gradient-based attribution | Medium (1 backward pass) | Optional signal |
| **Attention Rollout** | Attention flow | Low (from forward pass) | Attention entropy signal |
| **Feature Norm** | Activation magnitude | Free | Primary signal |
| **Frequency Content** | FFT of patch features | Low (from embeddings) | Primary signal |
| **Ablation-CAM** | Perturbation-based | High (N forward passes) | NOT used (too expensive) |

### Decision
- Primary signals: Feature Norm (free), Frequency Content (free), Attention Entropy (free)
- Optional signal: GradCAM (1 backward pass, useful for validation)
- NOT used: Ablation-CAM (too slow), RISE (too slow)

---

## 4. BACKBONE RECOMMENDATION

### Comparison

| Backbone | Pretraining | Public | Patch Features | Medical Adaptation |
|----------|------------|--------|---------------|-------------------|
| **DINOv2** | 142M images (natural) | ✅ torch.hub | ✅ Excellent | 🟡 Good (generalization) |
| **MedMAE** | Medical images | ✅ GitHub | ✅ Good | ✅ Medical-specific |
| **SAM** | 1B masks (natural) | ✅ Meta | ✅ Good | 🟡 Good |
| **RAD-DINO** | Radiology | ✅ HuggingFace | ✅ Good | ✅ Radiology |
| **CONCH** | Pathology | ✅ HuggingFace | ✅ Good | ✅ Pathology |
| **UNI** | Pathology | ✅ HuggingFace | ✅ Good | ✅ Pathology |

### Recommendation: DINOv2 as primary, MedMAE as secondary

**Rationale:**
1. DINOv2: Best general-purpose features, easy to load (torch.hub), excellent patch-level representation, works across all 3 modalities (skin, brain, pathology)
2. MedMAE: Medical-specific pretraining, may help for BRISC (brain MRI)
3. SAM: Good for segmentation but overkill for classification
4. Avoid RAD-DINO/CONCH/UNI: Single-modality (radiology/pathology only), doesn't generalize to our cross-modality setup

**Action**: Run both DINOv2 and MedMAE. Report DINOv2 as primary, MedMAE as "with medical pretraining" ablation.

---

## 5. SMALL LESION PROBLEM — Prior Work to Cite

1. **MFTMamba-Unet (2026)**: "Small-size medical lesions (<10% of image area) often lost during downsampling" — validates our problem
2. **npj Digital Medicine (2025)**: Rare lesions only 2-3% of annotations, up to 63.4% improvement with specialized methods — shows clinical importance
3. **EnFuseNet (2026)**: Long-tail skin lesion diagnosis — same modality (ISIC), validates class imbalance
4. **GAD-YOLO (2026)**: Diminutive/flat lesions in GI endoscopy often missed — validates detection challenge

**Our framing**: "Prior work addresses small lesions through specialized architectures or augmentation. We address it through token-level routing — a complementary and unders explored angle."

---

## 6. DATASET AVAILABILITY — Confirmed

| Dataset | Modality | Size | Download | License | Small Lesion Property |
|---------|----------|------|----------|---------|----------------------|
| **ISIC 2019** | Dermoscopy | 25,331 images | https://challenge.isic-archive.com/ | CC-0 / CC-BY | ✅ Melanomas can be tiny |
| **BRISC** | Brain MRI | 6,000 images | Kaggle/Figshare/Zenodo | CC BY 4.0 | ✅ Tumors vary in size |
| **MedMNIST v2** | Multi-organ | 708K (2D) + 10K (3D) | pip install medmnist | CC BY 4.0 | ✅ PathMNIST has cellular features |
| **PathMNIST** | Colon pathology | 107,180 | (subset of MedMNIST) | CC BY 4.0 | ✅ 9 tissue classes |

**Additional options:**
- **OCTMNIST** (retinal OCT, 109,309): Macular degeneration lesions
- **DermaMNIST** (7-class skin, 10,015): Lighter ISIC alternative
- **BloodMNIST** (blood cells, 17,092): Small cell-level features

**All confirmed publicly downloadable without restrictions.** ✅

---

## 7. AAAI FIT — Precedent Analysis

### Related AAAI 2025/2026 Acceptances

| Paper | Year | Domain | Relevance |
|-------|------|--------|-----------|
| TOP-RL | AAAI 2026 | General VLM token pruning | Shows AAAI accepts token efficiency papers |
| Recoverable Compression | AAAI 2025 | Text-guided token recovery | Token compression at AAAI |
| PPGPT | AAAI 2026 | Medical biosignal discrete tokens | Medical signal efficiency at AAAI |
| MedSpaformer | AAAI 2026 | Medical time series token sparsification | Medical token efficiency at AAAI |
| HeartLLM | AAAI 2026 | Medical ECG tokenization | Medical discrete representation at AAAI |

### AAAI Bar for Acceptance (Inferred)
- Clear problem statement with practical motivation ✅ (lesion preservation under compute budget)
- Technical novelty (not incremental) ✅ (multi-signal LATS is novel for medical)
- Thorough experiments (3+ datasets, multiple baselines) ⚠️ Need 3 datasets + 6 baselines
- Ablation studies ✅ (signal contribution, budget sweep)
- Analysis beyond accuracy (efficiency, qualitative) ✅ (lesion retention rate, visualization)

### What AAAI Reviewers Will Ask
1. "How is this different from frequency-aware token reduction (Lee 2025)?" → Multi-signal, medical-specific, lesion preservation metrics
2. "Why not use existing WSI pruning methods?" → Different scale (196 tokens vs thousands), different goal (classification vs slide-level)
3. "Is the medical motivation real or contrived?" → Need strong small-lesion analysis
4. "Does this generalize beyond the 3 tested datasets?" → Cross-modality experiment design is key

---

## 8. UPDATED RISK ASSESSMENT

| Risk | Level | Mitigation |
|------|-------|-----------|
| SparseLearn (Jun 2026) is very close conceptually | 🟡 Medium | Clearly differentiate: standard images vs WSI, lesion retention vs accuracy |
| Freq-Aware Token Reduction (Nov 2025) competes on "frequency" signal | 🟡 Medium | Multi-signal approach is richer; run as baseline; show medical benefit |
| Prompt-based pruning also uses ISIC | 🟡 Medium | They need prompts, we don't; they do segmentation, we do classification |
| Another group publishes similar work before Aug 2026 | 🟡 Medium | Post preprint on arXiv early (Week 3-4) |
| Reviewer says "just another token pruning paper" | 🟢 Low | Lesion-retention metric + medical motivation + cross-modality = differentiating |

---

## 9. FINAL VERDICT

**Direction is VIABLE and RECOMMENDED for AAAI 2027.**

The gap is confirmed: no paper does multi-signal lesion-aware token routing for standard medical image classification with lesion retention as a primary metric.

### Critical Adjustments to Our Approach

1. **Add Freq-Aware Token Reduction (Lee 2025) as a baseline** — it's the closest general method
2. **Clearly differentiate from WSI pruning** (SLIM, SparseLearn) — different scale, different problem
3. **Make lesion retention rate a PRIMARY metric** — not just an auxiliary analysis
4. **Use DINOv2 as primary backbone, MedMAE as ablation**
5. **Add PathMNIST for quick sanity checks, ISIC + BRISC for main results**
6. **Post preprint on arXiv by Week 3 of development**

### Recommended Baselines (Final List)
1. No pruning (upper bound)
2. Random pruning
3. ToMe (training-free, merging)
4. EViT (training-free, pruning)
5. DynamicViT (learned pruning)
6. Freq-Aware (Lee 2025) (frequency-based pruning)
7. **LATS (Ours)**

### Timeline Remains 8 Weeks ✅

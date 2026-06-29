"""
Configuration for MedTokenBudget: Lesion-Preserving Token Routing.

Target: AAAI 2027
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class RouterConfig:
    """Token routing configuration."""
    # Scoring signals
    use_attention_entropy: bool = True
    use_feature_norm: bool = True
    use_frequency_content: bool = True
    use_gradcam: bool = False          # Requires backward pass
    use_prediction_disagreement: bool = False

    # Router architecture
    router_hidden_dim: int = 128
    router_num_layers: int = 3
    router_dropout: float = 0.1

    # Budget
    token_budget_ratio: float = 0.5    # Fraction of patches to keep
    min_tokens: int = 16               # Minimum tokens regardless of budget
    spatial_smoothing_kernel: int = 3  # Spatial smoothing window

    # Training
    router_lr: float = 1e-3
    temperature: float = 1.0           # Gumbel-softmax temperature


@dataclass
class ModelConfig:
    """Backbone and task configuration."""
    backbone: Literal["dino_v2", "medmae", "sam", "resnet50"] = "dino_v2"
    backbone_size: Literal["small", "base", "large"] = "base"
    freeze_backbone: bool = True
    image_size: int = 224
    patch_size: int = 16

    # Task head
    num_classes: int = 8               # ISIC: 8, BRISC: 4, PathMNIST: 9
    head_type: Literal["linear", "mlp", "transformer"] = "mlp"
    head_hidden_dim: int = 256


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset: Literal["medmnist", "isic", "brisc", "all"] = "isic"
    data_dir: str = "./data"
    image_size: int = 224

    # ISIC
    isic_year: int = 2019

    # BRISC
    brisc_task: Literal["classification", "segmentation"] = "classification"

    # MedMNIST
    medmnist_subset: str = "pathmnist"  # pathmnist, dermamnist, octmnist, etc.

    # Training
    batch_size: int = 64
    num_workers: int = 4
    train_split: float = 0.8
    augment: bool = True


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    lr_scheduler: Literal["cosine", "step", "plateau"] = "cosine"
    warmup_epochs: int = 5

    # Loss weights
    cls_loss_weight: float = 1.0
    budget_reg_weight: float = 0.01
    lesion_loc_weight: float = 0.1     # Lesion localization auxiliary loss

    # Budget curriculum
    budget_curriculum: bool = True     # Gradually decrease budget during training
    budget_start: float = 1.0
    budget_end: float = 0.25
    budget_anneal_epochs: int = 30

    # Mixed precision
    use_amp: bool = True

    # Logging
    log_interval: int = 50
    eval_interval: int = 2
    save_best: bool = True


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    name: str = "med_token_budget"
    router: RouterConfig = field(default_factory=RouterConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Experiment tracking
    output_dir: str = "./results/med_token_budget"
    seed: int = 42
    device: str = "cuda"

    # Baseline comparisons
    baselines: List[str] = field(default_factory=lambda: [
        "no_pruning", "random", "dynamic_vit", "tome", "evit", "lats"
    ])


# Presets
ISIC_BASELINE = ExperimentConfig(
    name="isic_baseline",
    data=DataConfig(dataset="isic", isic_year=2019),
    model=ModelConfig(num_classes=8),
)

BRISC_BASELINE = ExperimentConfig(
    name="brisc_baseline",
    data=DataConfig(dataset="brisc"),
    model=ModelConfig(num_classes=4, backbone="medmae"),
)

MEDMNIST_QUICK = ExperimentConfig(
    name="medmnist_quick",
    data=DataConfig(dataset="medmnist", medmnist_subset="pathmnist"),
    model=ModelConfig(num_classes=9),
    train=TrainConfig(epochs=20),
)

BUDGET_SWEEP = ExperimentConfig(
    name="budget_sweep",
    router=RouterConfig(token_budget_ratio=0.5),
)

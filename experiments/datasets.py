"""
Data loaders for MedTokenBudget: MedMNIST, ISIC, BRISC.

All datasets are publicly available and preprocessed to 224×224.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# ─── Transforms ──────────────────────────────────────────────────────

def get_train_transforms(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_val_transforms(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ─── MedMNIST Dataset ────────────────────────────────────────────────

class MedMNISTDataset(Dataset):
    """
    Wrapper for MedMNIST v2 datasets. Lightweight 28×28 → 224×224 upsampled.

    Supports: pathmnist, dermamnist, octmnist, retinamnist, bloodmnist,
              tissuemnist, organamnist, organcmnist, organsmnist, etc.
    """

    SUBSETS = {
        'pathmnist': 'PathMNIST',
        'dermamnist': 'DermaMNIST',
        'octmnist': 'OCTMNIST',
        'pneumoniamnist': 'PneumoniaMNIST',
        'retinamnist': 'RetinaMNIST',
        'breastmnist': 'BreastMNIST',
        'bloodmnist': 'BloodMNIST',
        'tissuemnist': 'TissueMNIST',
        'organamnist': 'OrganAMNIST',
        'organcmnist': 'OrganCMNIST',
        'organsmnist': 'OrganSMNIST',
    }

    def __init__(
        self,
        subset: str = 'pathmnist',
        split: str = 'train',
        image_size: int = 224,
        download: bool = True,
        data_dir: str = './data',
        augment: bool = False,
    ):
        try:
            import medmnist
            from medmnist import INFO
        except ImportError:
            raise ImportError(
                "medmnist not installed. Run: pip install medmnist"
            )

        subset_key = self.SUBSETS.get(subset.lower(), subset)
        self.info = INFO[subset_key.lower()]

        DataClass = getattr(medmnist, subset_key)

        self.dataset = DataClass(
            split=split,
            download=download,
            root=data_dir,
            size=image_size,
        )

        self.image_size = image_size
        self.augment = augment and split == 'train'

        if augment:
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

        self.num_classes = len(self.info['label'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # MedMNIST returns numpy array, convert to PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.squeeze())

        # Convert to RGB if grayscale
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = self.transform(image)
        label = torch.tensor(label[0] if isinstance(label, np.ndarray) else label).long()

        return image, label


# ─── ISIC Dataset ────────────────────────────────────────────────────

class ISICDataset(Dataset):
    """
    ISIC Skin Lesion dataset.

    Download from: https://challenge.isic-archive.com/data/
    Supported years: 2018, 2019, 2020

    Expected directory structure:
        data/isic/
        ├── ISIC_2019_Training_Input/
        │   ├── ISIC_0000000.jpg
        │   └── ...
        ├── ISIC_2019_Training_GroundTruth.csv
        └── ...
    """

    def __init__(
        self,
        split: str = 'train',
        image_size: int = 224,
        data_dir: str = './data/isic',
        year: int = 2019,
        augment: bool = False,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.image_size = image_size
        self.data_dir = Path(data_dir)
        self.year = year
        self.augment = augment and split == 'train'

        # Load image paths and labels
        self.images, self.labels, self.lesion_masks = self._load_data()

        # Train/val split
        np.random.seed(seed)
        indices = np.random.permutation(len(self.images))
        split_idx = int(len(indices) * train_ratio)

        if split == 'train':
            indices = indices[:split_idx]
        elif split == 'val' or split == 'test':
            indices = indices[split_idx:]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.images = [self.images[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        if self.lesion_masks:
            self.lesion_masks = [self.lesion_masks[i] for i in indices]

        # Transforms
        if augment:
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

        self.num_classes = len(set(self.labels))
        logger.info(f"ISIC {year} {split}: {len(self.images)} images, {self.num_classes} classes")

    def _load_data(self):
        """Load ISIC data. Falls back to MedMNIST DermaMNIST if ISIC not found."""
        images = []
        labels = []
        masks = []

        # Try ISIC directory structure
        img_dir = self.data_dir / f"ISIC_{self.year}_Training_Input"
        gt_file = self.data_dir / f"ISIC_{self.year}_Training_GroundTruth.csv"

        if img_dir.exists() and gt_file.exists():
            import pandas as pd
            df = pd.read_csv(gt_file)
            image_col = 'image' if 'image' in df.columns else df.columns[0]

            for _, row in df.iterrows():
                img_name = row[image_col]
                if not img_name.endswith('.jpg'):
                    img_name += '.jpg'
                img_path = img_dir / img_name
                if img_path.exists():
                    images.append(str(img_path))
                    # Find label column
                    label_cols = [c for c in df.columns if c != image_col]
                    label = row[label_cols[0]]
                    if isinstance(label, str):
                        label = float(label)
                    labels.append(int(label))
                    masks.append(None)

        if not images:
            # Fallback: use MedMNIST DermaMNIST (subset of ISIC)
            logger.warning(f"ISIC {self.year} not found at {self.data_dir}. "
                          "Falling back to MedMNIST DermaMNIST.")
            from medmnist import DermaMNIST
            import medmnist

            dataset = DermaMNIST(split='train', download=True, root=str(self.data_dir.parent))
            logger.info(f"Loaded {len(dataset)} DermaMNIST samples as ISIC fallback")

            # Convert to image paths by saving to temp
            import tempfile
            tmp_dir = Path(tempfile.mkdtemp()) / "dermamnist_imgs"
            tmp_dir.mkdir(exist_ok=True)

            for i in range(len(dataset)):
                img, label = dataset[i]
                img_path = tmp_dir / f"derma_{i}.png"
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img.squeeze())
                img.save(img_path)
                images.append(str(img_path))
                labels.append(int(label[0] if isinstance(label, np.ndarray) else label))
                masks.append(None)

        return images, labels, masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(self.labels[idx]).long()

        # Lesion mask (if available, for retention evaluation)
        lesion_mask = None
        if idx < len(self.lesion_masks) and self.lesion_masks[idx] is not None:
            mask = Image.open(self.lesion_masks[idx]).convert('L')
            mask = T.Resize((self.image_size // 16, self.image_size // 16))(mask)
            mask = T.ToTensor()(mask).squeeze(0)
            lesion_mask = (mask > 0.5).float().flatten()  # [N]

        return image, label, lesion_mask


# ─── BRISC Dataset ───────────────────────────────────────────────────

class BRISCDataset(Dataset):
    """
    BRISC Brain Tumor MRI dataset.

    Download from Kaggle/Figshare/Zenodo (see paper).
    Expected structure:
        data/brisc/
        └── classification_task/
            ├── train/
            │   ├── glioma/
            │   ├── meningioma/
            │   ├── pituitary/
            │   └── no_tumor/
            └── test/
    """

    CLASSES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

    def __init__(
        self,
        split: str = 'train',
        image_size: int = 224,
        data_dir: str = './data/brisc',
        augment: bool = False,
    ):
        self.image_size = image_size
        self.data_dir = Path(data_dir)

        self.images = []
        self.labels = []

        task_dir = self.data_dir / 'classification_task' / split

        if task_dir.exists():
            for class_name in self.CLASSES:
                class_dir = task_dir / class_name
                if class_dir.exists():
                    for img_path in class_dir.glob('*.jpg'):
                        self.images.append(str(img_path))
                        self.labels.append(self.CLASS_TO_IDX[class_name])
                    for img_path in class_dir.glob('*.png'):
                        self.images.append(str(img_path))
                        self.labels.append(self.CLASS_TO_IDX[class_name])
            logger.info(f"BRISC {split}: {len(self.images)} images")
        else:
            logger.warning(f"BRISC not found at {task_dir}.")
            logger.info("Download from: https://www.kaggle.com/datasets/briscdataset/brisc2025/")

        self.num_classes = len(self.CLASSES)

        self.augment = augment and split == 'train'
        if augment:
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(self.labels[idx]).long()
        return image, label, None  # No lesion mask for BRISC


# ─── DataLoader Factory ──────────────────────────────────────────────

def get_dataloaders(config) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders based on config."""
    data_cfg = config.data
    train_cfg = config.train

    if data_cfg.dataset == 'medmnist':
        DatasetClass = MedMNISTDataset
        ds_kwargs = dict(
            subset=data_cfg.medmnist_subset,
            image_size=data_cfg.image_size or 224,
            data_dir=data_cfg.data_dir,
        )
    elif data_cfg.dataset == 'isic':
        DatasetClass = ISICDataset
        ds_kwargs = dict(
            image_size=data_cfg.image_size or 224,
            data_dir=data_cfg.data_dir,
            year=data_cfg.isic_year,
        )
    elif data_cfg.dataset == 'brisc':
        DatasetClass = BRISCDataset
        ds_kwargs = dict(
            image_size=data_cfg.image_size or 224,
            data_dir=data_cfg.data_dir,
        )
    else:
        raise ValueError(f"Unknown dataset: {data_cfg.dataset}")

    train_ds = DatasetClass(split='train', augment=data_cfg.augment, **ds_kwargs)
    val_ds = DatasetClass(split='val' if data_cfg.dataset != 'brisc' else 'test',
                          augment=False, **ds_kwargs)

    train_loader = DataLoader(
        train_ds, batch_size=data_cfg.batch_size,
        shuffle=True, num_workers=data_cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=data_cfg.batch_size * 2,
        shuffle=False, num_workers=data_cfg.num_workers,
        pin_memory=True,
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'num_classes': train_ds.num_classes,
        'train_size': len(train_ds),
        'val_size': len(val_ds),
    }

"""
Data loaders for MedTokenBudget: MedMNIST, ISIC, BRISC.

All datasets auto-download on first use — no manual steps needed.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
from PIL import Image
import logging
import zipfile
import shutil
import json
import urllib.request

logger = logging.getLogger(__name__)

# ─── Auto-Download Utilities ──────────────────────────────────────────

def _download_file(url: str, dest: Path, desc: str = "Downloading"):
    """Download a file with progress bar and resume support."""
    import urllib.request

    if dest.exists():
        logger.info(f"Already downloaded: {dest.name}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + '.part')

    # Check for partial download
    resume_pos = tmp_path.stat().st_size if tmp_path.exists() else 0

    logger.info(f"{desc} -> {dest.parent}")
    logger.info(f"  URL: {url}")

    try:
        req = urllib.request.Request(url)
        if resume_pos > 0:
            req.add_header('Range', f'bytes={resume_pos}-')
            logger.info(f"  Resuming from byte {resume_pos}")

        with urllib.request.urlopen(req) as response:
            total = int(response.headers.get('Content-Length', 0))
            mode = 'ab' if resume_pos > 0 else 'wb'

            with open(tmp_path, mode) as f:
                downloaded = resume_pos
                chunk_size = 1024 * 1024  # 1MB chunks

                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / (total + resume_pos) * 100
                        mb = downloaded / (1024 * 1024)
                        total_mb = (total + resume_pos) / (1024 * 1024)
                        print(f"\r  {desc}: {mb:.0f}/{total_mb:.0f} MB ({pct:.0f}%)", end='', flush=True)

        print()  # newline after progress
        tmp_path.rename(dest)
        logger.info(f"  Downloaded: {dest.name}")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info(f"  Partial download saved at {tmp_path} — will resume on next run")
        raise


def _extract_zip(zip_path: Path, dest_dir: Path, desc: str = "Extracting"):
    """Extract a zip file with progress reporting."""
    if dest_dir.exists() and any(dest_dir.iterdir()):
        logger.info(f"Already extracted: {dest_dir.name}")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{desc} -> {dest_dir}")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        for i, member in enumerate(members):
            zf.extract(member, dest_dir)
            if i % 100 == 0 or i == len(members) - 1:
                pct = (i + 1) / len(members) * 100
                print(f"\r  {desc}: {i+1}/{len(members)} files ({pct:.0f}%)", end='', flush=True)
    print()


def download_isic_2019(data_dir: Path) -> Path:
    """Auto-download ISIC 2019 dataset. Returns path to data."""
    isic_dir = data_dir / "isic" / "ISIC_2019"
    isic_dir.mkdir(parents=True, exist_ok=True)
    img_subdir = isic_dir / "ISIC_2019_Training_Input"

    # Fix previous double-nesting bug (must run before any early-return)
    double_nested = img_subdir / "ISIC_2019_Training_Input"
    if double_nested.exists() and any(double_nested.iterdir()):
        logger.info("Fixing double-nested extraction from previous run...")
        for item in double_nested.iterdir():
            shutil.move(str(item), str(img_subdir / item.name))
        double_nested.rmdir()

    # Check if images actually exist (look for JPEGs, not just marker)
    has_images = img_subdir.exists() and any(img_subdir.glob("*.jpg"))

    if has_images:
        logger.info(f"ISIC 2019 images ready ({len(list(img_subdir.glob('*.jpg')))} found)")
    else:
        # Download and extract
        img_zip = data_dir / "isic" / "ISIC_2019_Training_Input.zip"
        _download_file(
            "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip",
            img_zip, desc="ISIC 2019 images (9.3 GB)"
        )
        # Extract to isic_dir (zip has ISIC_2019_Training_Input/ internally)
        _extract_zip(img_zip, isic_dir, desc="Extracting ISIC images")
        img_zip.unlink(missing_ok=True)

    # Download ground truth CSV
    gt_path = isic_dir / "ISIC_2019_Training_GroundTruth.csv"
    if not gt_path.exists():
        _download_file(
            "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv",
            gt_path, desc="ISIC 2019 labels (2 MB)"
        )

    logger.info(f"ISIC 2019 ready at {isic_dir}")
    return isic_dir


def download_isic_2018_task1_masks(data_dir: Path) -> Optional[Path]:
    """Download ISIC 2018 Task 1 segmentation masks for retention analysis."""
    mask_dir = data_dir / "isic" / "ISIC2018_Task1_Training_GroundTruth"
    if mask_dir.exists() and any(mask_dir.glob("*_segmentation.png")):
        logger.info(f"ISIC 2018 segmentation masks ready ({len(list(mask_dir.glob('*_segmentation.png')))} found)")
        return mask_dir

    zip_path = data_dir / "isic" / "ISIC2018_Task1_Training_GroundTruth.zip"
    try:
        _download_file(
            "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip",
            zip_path,
            desc="ISIC 2018 Task1 masks (26 MB)",
        )
        logger.info(f"Extracting ISIC 2018 masks -> {data_dir / 'isic'}")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            members = zf.namelist()
            for i, member in enumerate(members):
                zf.extract(member, data_dir / "isic")
                if i % 100 == 0 or i == len(members) - 1:
                    pct = (i + 1) / len(members) * 100
                    print(
                        f"\r  Extracting ISIC 2018 masks: {i+1}/{len(members)} files ({pct:.0f}%)",
                        end='', flush=True,
                    )
        print()
        zip_path.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning(f"Could not download ISIC 2018 masks for retention analysis: {exc}")
        return None

    if mask_dir.exists() and any(mask_dir.glob("*_segmentation.png")):
        return mask_dir
    logger.warning(f"ISIC 2018 mask extraction did not produce masks under {mask_dir}")
    return None


def download_brisc(data_dir: Path) -> Path:
    """Auto-download BRISC brain tumor dataset from Figshare. Returns path."""
    brisc_dir = data_dir / "brisc"
    marker = brisc_dir / ".downloaded"

    def has_brisc_classes(path: Path) -> bool:
        task_dir = path / "classification_task"
        return (task_dir / "train").exists() and (task_dir / "test").exists()

    if marker.exists() and has_brisc_classes(brisc_dir):
        logger.info("BRISC already downloaded and extracted.")
        return brisc_dir
    if marker.exists():
        logger.warning("BRISC marker exists but extracted files are missing; redownloading.")
        marker.unlink(missing_ok=True)

    brisc_dir.mkdir(parents=True, exist_ok=True)

    def brisc_download_url() -> str:
        api_url = "https://api.figshare.com/v2/articles/30533120"
        try:
            with urllib.request.urlopen(api_url, timeout=30) as resp:
                article = json.loads(resp.read().decode("utf-8"))
            files = article.get("files", [])
            for file_info in files:
                if file_info.get("name", "").lower().endswith(".zip"):
                    return file_info["download_url"]
        except Exception as exc:
            logger.warning(f"Figshare API lookup failed; using known BRISC file URL: {exc}")
        return "https://ndownloader.figshare.com/files/59298329"

    zip_path = data_dir / "brisc2025.zip"
    if not has_brisc_classes(brisc_dir):
        if zip_path.exists() and not zipfile.is_zipfile(zip_path):
            logger.warning(f"Removing invalid BRISC archive: {zip_path}")
            zip_path.unlink()
        _download_file(
            brisc_download_url(),
            zip_path,
            desc="BRISC brain MRI (260 MB)"
        )
        if not zipfile.is_zipfile(zip_path):
            raise zipfile.BadZipFile(f"Downloaded BRISC archive is not a zip file: {zip_path}")
        _extract_zip(zip_path, brisc_dir, desc="Extracting BRISC")
        zip_path.unlink(missing_ok=True)

        # BRISC zip may have nested directory — flatten classification_task if needed.
        if not has_brisc_classes(brisc_dir):
            nested_task = next(
                (p for p in brisc_dir.rglob("classification_task") if p.is_dir()),
                None,
            )
            if nested_task and nested_task.parent != brisc_dir:
                target = brisc_dir / "classification_task"
                if target.exists():
                    shutil.rmtree(target)
                shutil.move(str(nested_task), str(target))
    else:
        logger.info("BRISC already extracted.")

    if not has_brisc_classes(brisc_dir):
        raise FileNotFoundError(
            f"BRISC extraction did not produce classification_task/train and test under {brisc_dir}"
        )

    marker.touch()
    logger.info(f"BRISC ready at {brisc_dir}")
    return brisc_dir


def download_with_fallback(download_func, data_dir: Path, fallback_name: str):
    """Try to download; fall back to MedMNIST on failure."""
    try:
        return download_func(data_dir)
    except Exception as e:
        logger.warning(f"Auto-download failed: {e}")
        logger.info(f"Falling back to MedMNIST {fallback_name}")
        return None


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
    """Wrapper for MedMNIST v2. Auto-downloads on first use."""

    SUBSETS = {
        'pathmnist': 'PathMNIST', 'dermamnist': 'DermaMNIST',
        'octmnist': 'OCTMNIST', 'pneumoniamnist': 'PneumoniaMNIST',
        'retinamnist': 'RetinaMNIST', 'breastmnist': 'BreastMNIST',
        'bloodmnist': 'BloodMNIST', 'tissuemnist': 'TissueMNIST',
        'organamnist': 'OrganAMNIST', 'organcmnist': 'OrganCMNIST',
        'organsmnist': 'OrganSMNIST',
    }

    def __init__(self, subset: str = 'pathmnist', split: str = 'train',
                 image_size: int = 224, download: bool = True,
                 data_dir: str = './data', augment: bool = False):
        try:
            import medmnist
            from medmnist import INFO
        except ImportError:
            raise ImportError("pip install medmnist")

        subset_key = self.SUBSETS.get(subset.lower(), subset)
        self.info = INFO[subset_key.lower()]
        DataClass = getattr(medmnist, subset_key)

        # Ensure root directory exists (MedMNIST requires it)
        root = Path(data_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)

        self.dataset = DataClass(split=split, download=download, root=str(root), size=image_size)
        self.image_size = image_size
        self.augment = augment and split == 'train'
        self.transform = get_train_transforms(image_size) if self.augment else get_val_transforms(image_size)
        self.num_classes = len(self.info['label'])

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.squeeze())
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        label = torch.tensor(label[0] if isinstance(label, np.ndarray) else label).long()
        return image, label


# ─── ISIC Dataset (with auto-download) ──────────────────────────────

class ISICDataset(Dataset):
    """ISIC 2019 Skin Lesion dataset. Auto-downloads on first use."""

    def __init__(self, split: str = 'train', image_size: int = 224,
                 data_dir: str = './data', year: int = 2019,
                 augment: bool = False, train_ratio: float = 0.8, seed: int = 42,
                 include_masks: bool = True, patch_grid_size: int = 16):
        self.image_size = image_size
        self.data_dir = Path(data_dir)
        self.year = year
        self.augment = augment and split == 'train'
        self.patch_grid_size = patch_grid_size
        self.include_masks = include_masks

        # Try auto-download; fall back to DermaMNIST
        isic_path = download_with_fallback(
            lambda d: download_isic_2019(d), self.data_dir, "DermaMNIST"
        )

        if isic_path and (isic_path / "ISIC_2019_Training_Input").exists():
            self._load_isic(isic_path, split, train_ratio, seed)
        else:
            self._load_dermamnist_fallback(split, seed)

        self.transform = get_train_transforms(image_size) if self.augment else get_val_transforms(image_size)
        self.num_classes = len(set(self.labels))
        logger.info(f"ISIC {split}: {len(self.images)} images, {self.num_classes} classes")

    def _load_isic(self, isic_path, split, train_ratio, seed):
        import pandas as pd

        img_dir = isic_path / "ISIC_2019_Training_Input"
        gt_file = isic_path / "ISIC_2019_Training_GroundTruth.csv"
        df = pd.read_csv(gt_file)
        image_index = {p.stem: p for p in img_dir.rglob("*.jpg")}
        mask_index = {}
        if self.include_masks:
            mask_dir = download_isic_2018_task1_masks(self.data_dir)
            if mask_dir is not None:
                mask_index = {
                    p.stem.replace("_segmentation", ""): p
                    for p in mask_dir.glob("*_segmentation.png")
                }

        images, labels, masks = [], [], []
        for _, row in df.iterrows():
            img_path = image_index.get(row['image'])
            if img_path is not None:
                images.append(str(img_path))
                # ISIC 2019 has one-hot columns: MEL, NV, BCC, AK, BKL, DF, VASC, SCC
                label_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
                label_vals = [row.get(c, 0) for c in label_cols]
                label = np.argmax(label_vals) if any(label_vals) else 0
                labels.append(label)
                mask_path = mask_index.get(row['image'])
                masks.append(str(mask_path) if mask_path is not None else None)
        if not images:
            raise RuntimeError(f"No ISIC images found under {img_dir}")

        np.random.seed(seed)
        indices = np.random.permutation(len(images))
        split_idx = int(len(indices) * train_ratio)
        idxs = indices[:split_idx] if split == 'train' else indices[split_idx:]

        self.images = [images[i] for i in idxs]
        self.labels = [labels[i] for i in idxs]
        self.lesion_masks = [masks[i] for i in idxs]

    def _load_dermamnist_fallback(self, split, seed):
        logger.info("Using DermaMNIST as ISIC fallback (skin lesion, 7 classes, 10K images)")
        from medmnist import DermaMNIST
        import tempfile

        dataset = DermaMNIST(split='train', download=True, root=str(self.data_dir))

        # Split locally
        np.random.seed(seed)
        indices = np.random.permutation(len(dataset))
        split_idx = int(len(indices) * 0.8)
        idxs = indices[:split_idx] if split == 'train' else indices[split_idx:]

        tmp_dir = Path(tempfile.mkdtemp()) / "dermamnist_imgs"
        tmp_dir.mkdir(exist_ok=True)

        images, labels = [], []
        for i in idxs:
            img, label = dataset[int(i)]
            img_path = tmp_dir / f"derma_{i}.png"
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img.squeeze())
            img.save(img_path)
            images.append(str(img_path))
            labels.append(int(label[0] if isinstance(label, np.ndarray) else label))

        self.images = images
        self.labels = labels
        self.lesion_masks = [None] * len(images)

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(self.labels[idx]).long()
        mask = None
        if idx < len(self.lesion_masks) and self.lesion_masks[idx] is not None:
            m = Image.open(self.lesion_masks[idx]).convert('L')
            m = T.Resize((self.patch_grid_size, self.patch_grid_size))(m)
            m = T.ToTensor()(m).squeeze(0)
            mask = (m > 0.5).float().flatten()
        if not self.include_masks:
            return image, label
        if mask is None:
            mask = torch.zeros(self.patch_grid_size * self.patch_grid_size)
            mask_valid = torch.tensor(0.0)
        else:
            mask_valid = torch.tensor(1.0)
        return image, label, mask, mask_valid


# ─── BRISC Dataset (with auto-download) ─────────────────────────────

class BRISCDataset(Dataset):
    """BRISC Brain Tumor MRI. Auto-downloads from Figshare on first use."""

    CLASSES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

    def __init__(self, split: str = 'train', image_size: int = 224,
                 data_dir: str = './data', augment: bool = False):
        self.image_size = image_size
        self.data_dir = Path(data_dir)
        self.augment = augment and split == 'train'

        # Auto-download
        brisc_path = download_with_fallback(
            lambda d: download_brisc(d), self.data_dir, "PathMNIST"
        )

        self.images, self.labels = [], []

        task_dir = None
        if brisc_path:
            task_dir = brisc_path / 'classification_task' / split

        if task_dir and task_dir.exists():
            for class_name in self.CLASSES:
                class_dir = task_dir / class_name
                if class_dir.exists():
                    for ext in ['*.jpg', '*.png', '*.jpeg']:
                        for img_path in class_dir.glob(ext):
                            self.images.append(str(img_path))
                            self.labels.append(self.CLASS_TO_IDX[class_name])
            logger.info(f"BRISC {split}: {len(self.images)} images")
        else:
            logger.warning(f"BRISC not found. Run again to retry download.")
            logger.info("  Resume support: partial downloads will continue from breakpoint.")

        self.num_classes = len(self.CLASSES)
        self.transform = get_train_transforms(image_size) if self.augment else get_val_transforms(image_size)

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(self.labels[idx]).long()
        return image, label


# ─── DataLoader Factory ──────────────────────────────────────────────

def get_dataloaders(config) -> Dict[str, DataLoader]:
    """Create train/val dataloaders. All datasets auto-download."""
    data_cfg = config.data

    # Resolve to absolute path (avoids relative-path issues on different systems)
    data_dir = str(Path(data_cfg.data_dir).resolve())
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    effective_patch_size = 14 if config.model.backbone == "dino_v2" else config.model.patch_size
    patch_grid_size = max(1, (data_cfg.image_size or 224) // effective_patch_size)

    ds_map = {
        'medmnist': (MedMNISTDataset, dict(
            subset=data_cfg.medmnist_subset,
            image_size=data_cfg.image_size or 224,
            data_dir=data_dir,
        )),
        'isic': (ISICDataset, dict(
            image_size=data_cfg.image_size or 224,
            data_dir=data_dir,
            year=data_cfg.isic_year,
            include_masks=data_cfg.include_lesion_masks,
            patch_grid_size=patch_grid_size,
        )),
        'brisc': (BRISCDataset, dict(
            image_size=data_cfg.image_size or 224,
            data_dir=data_dir,
        )),
    }

    if data_cfg.dataset not in ds_map:
        raise ValueError(f"Unknown dataset: {data_cfg.dataset}. Choose: {list(ds_map.keys())}")

    DatasetClass, ds_kwargs = ds_map[data_cfg.dataset]
    val_split = 'test' if data_cfg.dataset == 'brisc' else 'val'

    train_ds = DatasetClass(split='train', augment=data_cfg.augment, **ds_kwargs)
    val_ds = DatasetClass(split=val_split, augment=False, **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=data_cfg.batch_size,
                              shuffle=True, num_workers=data_cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=data_cfg.batch_size * 2,
                            shuffle=False, num_workers=data_cfg.num_workers, pin_memory=True)

    return {
        'train': train_loader, 'val': val_loader,
        'num_classes': train_ds.num_classes,
        'train_size': len(train_ds), 'val_size': len(val_ds),
    }

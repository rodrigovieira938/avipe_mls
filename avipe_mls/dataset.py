from .types import DatasetConfig
import kagglehub
import huggingface_hub
import importlib.util
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np
import torch
import albumentations as A

ImageFile.LOAD_TRUNCATED_IMAGES = True

class DatasetDownloader:
    def __init__(self, config: DatasetConfig):
        self.config = config

    def download(self):
        source_type = self.config.source.type
        path = ""
        if source_type == "kaggle":
            path = self._download_kaggle()
        elif source_type == "huggingface":
            path = self._download_huggingface()
        else:
            raise ValueError(f"Unknown dataset source type: {source_type}")
        self._run_post_download_scripts(path)
        return str(path)
    def _download_kaggle(self):
        return kagglehub.download_competition(self.config.source.competition) # type: ignore

    def _download_huggingface(self):
        return huggingface_hub.snapshot_download(repo_id=self.config.source.repo, repo_type=self.config.source.repo_type) # type: ignore
    def _run_post_download_scripts(self, dataset_path: str):
        for script_path in self.config.post_download_scripts:
            script_path_obj = Path(script_path)
            if not script_path_obj.is_file():
                raise FileNotFoundError(f"Post-download script not found: {script_path}")
            spec = importlib.util.spec_from_file_location("post_download_script", script_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load script: {script_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, "run"):
                raise AttributeError(f"Script {script_path} does not have a 'run' function")
            print(f"Running post-download script: {script_path}")
            module.run(dataset_path)

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]

class SegmentationDataset(Dataset):
    def __init__(self, images_dir: str | Path, masks_dir: str | Path, transform=A.Compose([A.Resize(256, 256),A.pytorch.ToTensorV2()])):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        # Collect all image files with supported extensions
        self.images = sorted(
            [p for p in self.images_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        )
        self.masks = sorted(
            [p for p in self.masks_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        )

        if len(self.images) != len(self.masks):
            raise ValueError("Number of images and masks do not match!")
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert("RGB"))
        mask = np.array(Image.open(self.masks[idx]).convert("L"), dtype=np.int64)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        return image.float(), mask

def create_dataset(config: DatasetConfig) -> SegmentationDataset:
    path = Path(DatasetDownloader(config).download())
    images_dir = Path.joinpath(path, config.images_path)
    masks_dir = Path.joinpath(path, config.masks_path)
    return SegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
    )
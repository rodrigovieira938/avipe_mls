import kagglehub
from ._dataset import FolderDataset, Dataset
from PIL import Image
from torchvision import transforms
import huggingface_hub
import os
import tarfile
from pathlib import Path
import numpy as np
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CarvanaDataset(Dataset):
    def __init__(self) -> None:
        self.dataset = None
        self.transform =  transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()])
    def ensure(self):
        dataset_path = kagglehub.competition_download("carvana-image-masking-challenge")
        self.dataset = FolderDataset(dataset_path, transform=self.transform)
    def __getitem__(self, index):
        if self.dataset == None:
            self.ensure()
            assert self.dataset is not None
        return self.dataset[index]

    def __len__(self):
        if self.dataset == None:
            self.ensure()
            assert self.dataset is not None
        return len(self.dataset)

class GrapevistaVitigeossDataset(Dataset):
    def __init__(self) -> None:
        self.dataset = None
        self.transform =  transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()])
        self.dataset_path = ""
        self.images = None
    def __concat_parts(self):
        parts = sorted(Path(f"{self.dataset_path}/data/").glob("grapevista.tar.*.gz.part"))  # Sort to ensure correct order
        print("Concatting parts")
        with open(f"{self.dataset_path}/data/grapevista.tar.gz", 'wb') as f_out:
            for part in parts:
                with open(part, 'rb') as f_in:
                    while chunk := f_in.read(1024 * 1024):
                        f_out.write(chunk)

    def __extract_tar_gz(self):
        print("Decompressing")
        with tarfile.open(f"{self.dataset_path}/data/grapevista.tar.gz", 'r:gz') as tar:
            tar.extractall(path=f"{self.dataset_path}/data")
    #TODO: Check for truncated images and remove them from self.images and self.masks
    def ensure(self):
        self.dataset_path = huggingface_hub.snapshot_download(repo_id="links-ads/grapevista-dataset", repo_type="dataset")
        print("Grapevista Dataset is at",self.dataset_path)
        if not os.path.exists(f"{self.dataset_path}/data/grapevista"):
            self.__concat_parts()
            self.__extract_tar_gz()
        self.root_dir = f"{self.dataset_path}/data/grapevista/vitigeoss/"
        
        self.image_dir = self.root_dir + "/images"
        self.masks_dir = self.root_dir + "/annotations"
        
        self.images = []
        self.masks = []
        
        for img in os.listdir(self.image_dir):
            mask_dir = os.path.join(self.masks_dir, f"{img[0:-4]}.png")
            if not os.path.exists(mask_dir):
                continue
            self.images.append(os.path.join(self.image_dir, img))
            self.masks.append(mask_dir)
    def __getitem__(self, index):
        if self.images == None:
            self.ensure()
            assert self.images is not None
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")
        img, mask = self.transform(img), self.transform(mask).long().squeeze(0)
        return img, mask
    def __len__(self):
        if self.images == None:
            self.ensure()
            assert self.images is not None
        return len(self.images)
import os
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision import transforms
from PIL import Image

class Dataset(TorchDataset):
    def __init__(self):
        super().__init__()
    def __getitem__(self, index):
        return None
    def __len__(self):
        return 0
class FolderDataset(Dataset):
    def __init__(self, root_dir:str, image_dir: str = "train", masks_dir: str = "train_masks", transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_dir)
        self.masks_dir = os.path.join(root_dir, masks_dir)
        self.transform = transform if transform else transforms.ToTensor()
        self.images = []
        self.masks = []
        self.images = sorted([os.path.join(self.image_dir, i) for i in os.listdir(self.image_dir)])
        self.masks = sorted([os.path.join(self.masks_dir, i) for i in os.listdir(self.masks_dir)])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")
        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.images)

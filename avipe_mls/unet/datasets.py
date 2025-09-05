import kagglehub
from ._dataset import FolderDataset, Dataset
from PIL import Image
from torchvision import transforms

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
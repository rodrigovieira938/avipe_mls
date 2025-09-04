import torch.nn as nn
from pathlib import Path
import torch

class Checkpoint:
    def __init__(self, model: nn.Module, epoch:int, train_loss: float, val_loss: float, val_iou: float, vals_iou: 'list[float]', vals_loss: 'list[float]', train_losses: 'list[float]') -> None:
        self.format_version = 1 # Put this up when the format changes. TODO:Handle loading old versions in the load method.
        self.uuid = model.uuid
        self.model = model.state_dict()
        self.val_loss = val_loss
        self.train_loss = train_loss
        self.val_iou = val_iou
        self.epoch = epoch
        self.vals_iou = vals_iou
        self.vals_loss = vals_loss
        self.train_losses = train_losses
    def save(self, filepath):
        torch.save(self, filepath)
    @staticmethod
    def load(filepath) -> 'Checkpoint | None':
        checkpoint: Checkpoint = torch.load(filepath, weights_only=False)
        if not isinstance(checkpoint, Checkpoint):
            print(f"{filepath} is not a valid checkpoint")
            return None
        return checkpoint
class CheckpointManager:
    def __init__(self, root_path: str) -> None:
        self.root_path = root_path
    def save(self,model_name:str, checkpoint: Checkpoint, tag:str = "", latest: bool = True):
        savepath = Path(f"{self.root_path}/{model_name}")
        filename = f"{model_name}{f'-{tag}' if tag else ''}-{checkpoint.epoch}.pth"
        filepath = f"{savepath}/{filename}"
        savepath.mkdir(parents=True, exist_ok=True)
        if(latest):
            with open(f"{savepath}/latest.link", "w") as latest_link:
                latest_link.write(filename)
                latest_link.close()
        checkpoint.save(filepath)
    def get_latest(self, model_name: str) -> Checkpoint | None:
        savepath = Path(f"{self.root_path}/{model_name}")
        latest_link_path = savepath / "latest.link"
        if latest_link_path.exists():
            with open(latest_link_path, "r") as latest_link:
                filepath = latest_link.read().strip()
                latest_link.close()
                full_path = savepath / filepath
                if full_path.exists():
                    return Checkpoint.load(full_path)
        return None
    def get(self, model_name: str, checkpoint_name: str) -> Checkpoint | None:
        savepath = Path(f"{self.root_path}/{model_name}")
        checkpoint_path = savepath / checkpoint_name
        if checkpoint_path.exists():
            return Checkpoint.load(checkpoint_path)
        return None
    def get_all(self, model_name: str) -> 'list[Checkpoint]':
        savepath = Path(f"{self.root_path}/{model_name}")
        checkpoints = []
        if savepath.exists():
            for file in savepath.glob(f"{model_name}-*.pth"):
                checkpoints.append(Checkpoint.load(file))
        return checkpoints
    def get_latest_filepath(self, model_name: str) -> str | None:
        savepath = Path(f"{self.root_path}/{model_name}")
        latest_link_path = savepath / "latest.link"
        if latest_link_path.exists():
            with open(latest_link_path, "r") as latest_link:
                filepath = latest_link.read().strip()
                latest_link.close()
                return str(filepath)
        return None
    def get_all_filepaths(self, model_name: str) -> 'list[str]':
        savepath = Path(f"{self.root_path}/{model_name}")
        checkpoints = []
        if savepath.exists():
            for file in savepath.glob(f"{model_name}-*.pth"):
                checkpoints.append(str(file.relative_to(savepath)))
        return checkpoints
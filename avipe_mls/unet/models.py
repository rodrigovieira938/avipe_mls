#Maybe make this more generic and move to the main module
from ._model import UNet
from .._checkpoint import Checkpoint, CheckpointManager
from ._dataset import Dataset
from . import datasets
checkpoint_manager = CheckpointManager("weights")

class _ModelManager:
    def __init__(self, name:str, dataset:Dataset, num_classes=1):
        self.name = name
        self.checkpoints = checkpoint_manager.get_all_filepaths(name)
        self.latest = checkpoint_manager.get_latest_filepath(name)
        self.dataset = dataset
        self.num_classes = num_classes
        print(f"Latest model for {name}: {self.latest}")
    def load_latest(self) -> UNet | None:
        if self.latest:
            model = UNet(in_channels=3, num_classes=self.num_classes)
            checkpoint = checkpoint_manager.get(self.name, self.latest)
            if checkpoint:
                model.load_state_dict(checkpoint.model)
                return model
        return None
    def load(self, name:str) -> UNet | None:
        model = UNet(in_channels=3, num_classes=self.num_classes)
        if(name.upper() == "LATEST"):
            return self.load_latest()
        if(name.upper() == "NONE"):
            return model # Start from scratch
        checkpoint = checkpoint_manager.get(self.name, name)
        if checkpoint:
            model.load_state_dict(checkpoint.model)
            return model
        return None
    def get_latest(self) -> Checkpoint | None:
        if self.latest:
            checkpoint = checkpoint_manager.get(self.name, self.latest)
            return checkpoint
        return None
    def get(self, name:str) -> Checkpoint | None:
        if(name.upper() == "LATEST"):
            return checkpoint_manager.get_latest(self.name)
        if(name.upper() == "NONE"):
            return Checkpoint(self.load("none"), 0, 0, 0, 0, [], [], []) # type: ignore
        return checkpoint_manager.get(self.name, name)

def load_model(path:str, in_channels:int=3, num_classes:int=1):
    model = UNet(in_channels=in_channels, num_classes=num_classes)
    checkpoint = checkpoint_manager.get_latest(path)
    if checkpoint:
        model.load_state_dict(checkpoint.model)
        return model
    return None

_ModelDatabase = {
    "Carvana": _ModelManager("carvana", datasets.CarvanaDataset()),
}

def GetModel(name):
    return _ModelDatabase.get(name)
def SearchByPrefix(name):
    return [model for model in _ModelDatabase.keys() if model.startswith(name)]
def SearchBySuffix(name):
    return [model for model in _ModelDatabase.keys() if model.endswith(name)]
def GetModelNames():
    return list(_ModelDatabase.keys())
def GetModels():
    return _ModelDatabase
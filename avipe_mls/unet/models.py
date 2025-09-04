#Maybe make this more generic and move to the main module
from ._model import UNet
from .._checkpoint import CheckpointManager

checkpoint_manager = CheckpointManager("weights")

class _ModelManager:
    def __init__(self, name:str):
        self.name = name
        self.checkpoints = checkpoint_manager.get_all_filepaths(name)
        self.latest = checkpoint_manager.get_latest_filepath(name)
        print(f"Latest model for {name}: {self.latest}")
    def load_latest(self) -> UNet | None:
        if self.latest:
            model = UNet(in_channels=3, num_classes=1)
            checkpoint = checkpoint_manager.get(self.name, self.latest)
            if checkpoint:
                model.load_state_dict(checkpoint.model)
                return model
        return None
    def load(self, name:str) -> UNet | None:
        model = UNet(in_channels=3, num_classes=1)
        checkpoint = checkpoint_manager.get(self.name, name)
        if checkpoint:
            model.load_state_dict(checkpoint.model)
            return model
        return None

def load_model(path:str, in_channels:int=3, num_classes:int=1):
    model = UNet(in_channels=in_channels, num_classes=num_classes)
    checkpoint = checkpoint_manager.get_latest(path)
    if checkpoint:
        model.load_state_dict(checkpoint.model)
        return model
    return None

_ModelDatabase = {
    "Carvana": _ModelManager("carvana"),
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
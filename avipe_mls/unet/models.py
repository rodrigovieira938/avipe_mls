from ._model import UNet
from .._checkpoint import CheckpointManager
import os

checkpoint_manager = CheckpointManager("weights")

def load_model(path:str, in_channels:int=3, num_classes:int=1):
    model = UNet(in_channels=in_channels, num_classes=num_classes)
    checkpoint = checkpoint_manager.get_latest(path)
    if checkpoint:
        model.load_state_dict(checkpoint.model)
        return model
    return None

_ModelDatabase = {
    "carvana": load_model("carvana")
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
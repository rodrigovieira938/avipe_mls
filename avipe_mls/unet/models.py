from ._model import UNet
import os

carvana_model = None
if os.path.exists("weights/carvana-unet.pth"):
    carvana_model = UNet(in_channels=3, num_classes=1)
    carvana_model.load("weights/carvana-unet.pth")

_ModelDatabase = {
    "Carvana": carvana_model
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
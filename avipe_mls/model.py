import torch
from avipe_mls.types import FullConfig
import segmentation_models_pytorch as smp

class ModelWrapper(torch.nn.Module):
    def __init__(self, config:FullConfig,model_index:int, model: torch.nn.Module):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.name = config.name + "-" + config.model[model_index].backend
        self.config = config
        self.model_index = model_index
        self.model_config = config.model[model_index]

    def forward(self, x):
        return self.model(x)
    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
        return self
    def save(self, path: str, epoch: int | None = None):
        filepath = f"{path}/{self.name}" + (f"_{epoch}.pth" if epoch is not None else ".pth")
        torch.save(self.model.state_dict(), filepath)

def load_model(config: FullConfig, model_index:int = 0) -> ModelWrapper:
    model_config = config.model[model_index]
    if model_config.backend.lower() == "unet":
        model = smp.Unet(
            encoder_name=model_config.backbone,
            in_channels=model_config.in_channels,
            classes=config.dataset.num_classes,
            encoder_weights=None
        )
        return ModelWrapper(config,model_index, model)
    elif model_config.backend.lower() == "deeplabv3":
        model = smp.DeepLabV3(
            encoder_name=model_config.backbone,
            in_channels=model_config.in_channels,
            classes=config.dataset.num_classes,
            encoder_weights=None
        )
        return ModelWrapper(config,model_index, model)
    elif model_config.backend.lower() == "deeplabv3+":
        model = smp.DeepLabV3Plus(
            encoder_name=model_config.backbone,
            in_channels=model_config.in_channels,
            classes=config.dataset.num_classes,
            encoder_weights=None
        )
        return ModelWrapper(config,model_index, model)
    elif model_config.backend.lower() == "fpn":
        model = smp.FPN(
            encoder_name=model_config.backbone,
            in_channels=model_config.in_channels,
            classes=config.dataset.num_classes,
            encoder_weights=None
        )
        return ModelWrapper(config,model_index, model)
    else:
        raise ValueError(f"Unknown model name: {model_config.backend}")
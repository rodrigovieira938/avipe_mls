import torch
from avipe_mls.types import ModelConfig, DatasetConfig
import segmentation_models_pytorch as smp

def load_model(model_config: ModelConfig, dataset_config: DatasetConfig) -> torch.nn.Module:
    if model_config.name.lower() == "unet":
        model = smp.Unet(
            encoder_name=model_config.backbone,
            in_channels=model_config.in_channels,
            classes=dataset_config.num_classes
        )
        return model
    else:
        raise ValueError(f"Unknown model name: {model_config.name}")
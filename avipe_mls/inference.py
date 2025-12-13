from typing import Any, Mapping
import numpy as np
import torch
import albumentations as A

from . import model
from . import types

def run_inference(model_config: types.ModelConfig, dataset_config: types.DatasetConfig, weights : Mapping[str, Any], input_data) -> torch.Tensor:
    m = model.load_model(model_config, dataset_config)
    m.load_state_dict(weights)
    m.eval()
    with torch.no_grad():
        transform =  A.Compose([
                A.Resize(model_config.input_size[0], model_config.input_size[1]),
                A.pytorch.ToTensorV2()
        ])
        input = None
        output = m(transform(image=input_data)["image"].float().unsqueeze(0))
        pred_mask = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
    return pred_mask

def create_colored_mask(pred_mask: torch.Tensor, colors: np.typing.NDArray[np.uint8]):
    height, width = pred_mask.shape

    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_idx, color in enumerate(colors):
        colored_mask[pred_mask == class_idx] = color
    return colored_mask
from typing import Any, Mapping
import numpy as np
import torch
import albumentations as A

from . import model

def run_inference(model: model.ModelWrapper, input_data) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        transform =  A.Compose([
                A.Resize(model.model_config.input_size[0], model.model_config.input_size[1]),
                A.pytorch.ToTensorV2()
        ])
        output = model(transform(image=input_data)["image"].float().unsqueeze(0))
        pred_mask = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
    return pred_mask

def create_colored_mask(pred_mask: torch.Tensor, colors: np.typing.NDArray[np.uint8]):
    height, width = pred_mask.shape

    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_idx, color in enumerate(colors):
        colored_mask[pred_mask == class_idx] = color
    return colored_mask
def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5):
    return (image * (1 - alpha) + mask * alpha).astype(np.uint8)
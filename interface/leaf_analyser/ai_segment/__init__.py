from .post_process import postprocess_3class

from PIL import Image
import numpy as np
import torch
import albumentations as A

def rgb_to_rgba(rgb: np.ndarray, alpha_value=255) -> np.ndarray:
    h, w, _ = rgb.shape
    
    alpha = np.full((h, w, 1), alpha_value, dtype=rgb.dtype)
    rgba = np.concatenate((rgb, alpha), axis=2)
    
    return rgba
def remove_background(image: np.ndarray, mask: np.ndarray, background_color=(0, 0, 0, 0)):
    result = rgb_to_rgba(image.copy())
    result[mask == 0] = background_color
    return result


def segment_image(image, pt_path):
    input_data = np.array(image.resize((800, 800)))
    # Load the model
    model = torch.jit.load(pt_path)

    # Preprocess the image (resize, normalize, etc.)
    transform = A.Compose([
        A.pytorch.ToTensorV2()
    ])
    input_tensor = transform(image=input_data)["image"].float().unsqueeze(0)
    output = model(input_tensor)
    pred_mask = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
    pred_mask_np = pred_mask.numpy()
    pred_mask_np = postprocess_3class(
        pred_mask_np,
        background_id=0,
        class_ids=(1, 2),
        priority=(2, 1, 0),
        close_k=5,
        min_area=200
    )

    return remove_background(input_data, pred_mask_np)

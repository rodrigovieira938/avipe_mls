from PIL import Image
import numpy as np
from skimage.color import rgb2lab
from scipy.ndimage import binary_fill_holes


def resegment(img):
    rgba = np.array(img)
    rgb = rgba[:, :, :3] / 255.0
    alpha = rgba[:, :, 3]

    # Convert to Lab
    lab = rgb2lab(rgb)  # L in [0,100], a and b roughly [-128,127]

    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    L_min, L_max = 20, 80
    A_min, A_max = -20, 20
    B_min, B_max = 10, 60

    mask = (
        (L >= L_min) & (L <= L_max) &
        (A >= A_min) & (A <= A_max) &
        (B >= B_min) & (B <= B_max)
    )

    mask_filled = binary_fill_holes(mask)

    output = np.zeros_like(rgba)
    output[mask_filled] = rgba[mask_filled]
    return output
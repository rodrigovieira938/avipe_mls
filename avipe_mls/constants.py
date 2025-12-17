import torch

DEFAULT_ROOT_PATH = "experiments-output"
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
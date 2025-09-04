from typing import Tuple
import torch
import cpuinfo

def get_cpu_name() -> str:
    info = cpuinfo.get_cpu_info()
    return info.get('brand_raw', 'Unknown CPU')
def query_devices() -> dict[str, str]:
    devices = {}
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_properties(i).name
            devices[f'{name}'] = f'cuda:{i}'
    devices[get_cpu_name()] = 'cpu'
    return devices
def can_run_batch(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    input_shape: Tuple[int, ...]
) -> bool:
    model.eval()
    dummy_input = torch.randn((batch_size, *input_shape), device=device)
    try:
        with torch.no_grad():
            _ = model(dummy_input)
        return True
    except RuntimeError as e:
        if 'out of memory' in str(e):
            torch.cuda.empty_cache()
            return False
        else:
            raise e

def find_max_batch_size(
    model: torch.nn.Module,
    device: torch.device,
    input_shape: Tuple[int, ...],
    max_search: int = 1024
) -> int:
    low, high = 1, max_search
    best = 0

    while low <= high:
        mid = (low + high) // 2
        if can_run_batch(model, device, mid, input_shape):
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best
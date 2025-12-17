from . import types
from . import constants

def _get_save_path(full_config: types.FullConfig, root_path:str=constants.DEFAULT_ROOT_PATH):
    return f"{root_path}/{full_config.name}"
def _get_pth_path(full_config: types.FullConfig, model_idx: int, epoch_number: int | None = None, root_path:str=constants.DEFAULT_ROOT_PATH):
    save_path = _get_save_path(full_config, root_path)
    return f"{save_path}/{full_config.model[model_idx].name}_{epoch_number}.pth"
def _get_csv_path(full_config: types.FullConfig, model_idx: int, root_path:str=constants.DEFAULT_ROOT_PATH):
    save_path = _get_save_path(full_config, root_path)
    return f"{save_path}/{full_config.model[model_idx].name}.csv"
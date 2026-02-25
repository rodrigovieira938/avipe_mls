import argparse
from pathlib import Path
import torch

from .. import config
from .. import model
from .. import utils
from .. import constants


def _find_latest_epoch(full_config, model_idx: int, root_path: str) -> int | None:
    """Return the highest epoch number that has a saved weight file for the
    given model, or ``None`` if none are found.
    """
    max_epoch = full_config.get_training_for_model(model_idx).epochs
    for epoch in range(max_epoch, 0, -1):
        pth = utils._get_pth_path(full_config, model_idx, epoch, root_path)
        if Path(pth).is_file():
            return epoch
    return None


def export_torchscript(
    full_config: config.FullConfig,
    model_idx: int,
    epoch_number: int,
    root_path: str = constants.DEFAULT_ROOT_PATH,
    output_path: str | None = None,
) -> str:
    """Load a model checkpoint and export it as a torchscript file.

    Returns the path to the generated file.
    """

    pth_path = utils._get_pth_path(full_config, model_idx, epoch_number, root_path)
    if not Path(pth_path).is_file():
        raise FileNotFoundError(f"No weights file found at {pth_path}")

    wrapper = model.load_model(full_config, model_idx)
    wrapper.load(pth_path)
    wrapper.eval()
    wrapper.cpu()

    # determine size from config
    model_cfg = full_config.model[model_idx]
    in_ch = model_cfg.in_channels
    width, height = model_cfg.input_size

    dummy = torch.randn(1, in_ch, height, width)
    traced = torch.jit.trace(wrapper, dummy)

    if output_path is None:
        base = utils._get_save_path(full_config, root_path)
        output_path = f"{base}/{model_cfg.name}.pt"

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_file))
    return str(out_file)


def _parse_model_selection(model_str: str, full_config: config.FullConfig) -> int:
    """Convert a comma separated list of model identifiers to a single index.

    Only one model is allowed; an error is raised if the string resolves to
    zero or multiple indexes.
    """
    candidates: list[int] = []
    if model_str.lower() == "all":
        candidates = list(range(len(full_config.model)))
    else:
        for token in model_str.split(","):
            token = token.strip()
            for idx, mcfg in enumerate(full_config.model):
                if mcfg.name.lower() == token.lower():
                    candidates.append(idx)
                    break
            else:
                raise ValueError(f"Model not found inside config: \"{token}\"")
    if len(candidates) != 1:
        raise ValueError("Export tool requires exactly one model to be specified")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Avipe mls export tool",
        description="Export a trained model checkpoint as a torchscript file (single model/epoch)",
    )
    parser.add_argument("-c", "--config", help="Path to config.yaml file", required=True)
    parser.add_argument("-m", "--model", help="Model to export or 'all'", default="all")
    parser.add_argument(
        "-r",
        "--root-path",
        help="Root path to experiment weights directory",
        default=constants.DEFAULT_ROOT_PATH,
    )
    parser.add_argument(
        "-e",
        "--epoch",
        type=int,
        help="Epoch number to export. If not provided the most recent available epoch is used.",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output-path",
        help="Where to write the torchscript file. Defaults to the experiment directory with '.pt' extension",
        default=None,
    )
    args = parser.parse_args()

    full_cfg = config.load(args.config)
    try:
        model_idx = _parse_model_selection(args.model, full_cfg)
    except ValueError as exc:
        print(exc)
        exit(1)

    # determine epoch
    epoch_num = args.epoch
    if epoch_num is None:
        found = _find_latest_epoch(full_cfg, model_idx, args.root_path)
        if found is None:
            print(f"Didn't find any weights for {full_cfg.model[model_idx].name}")
            exit(1)
        epoch_num = found
    else:
        pth = utils._get_pth_path(full_cfg, model_idx, epoch_num, args.root_path)
        if not Path(pth).is_file():
            print(f"Didn't find weights for epoch {epoch_num} for {full_cfg.model[model_idx].name}")
            exit(1)

    out = export_torchscript(full_cfg, model_idx, epoch_num, args.root_path, args.output_path)
    print(f"Exported model to {out}")


if __name__ == "__main__":
    main()

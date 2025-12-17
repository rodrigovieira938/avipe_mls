import argparse
from pathlib import Path
from PIL import Image
import numpy as np

from .. import config
from .. import model
from .. import inference as _inference
from .. import constants
from .. import utils

def inference(input_file, model_indexes, epoch_numbers, full_config:config.FullConfig, root_path=constants.DEFAULT_ROOT_PATH):
    masks = []

    img = Image.open(input_file).convert("RGB")
    img_np = np.array(img)
    for idx, model_idx in enumerate(model_indexes):
        epoch = epoch_numbers[idx]
        model_path = utils._get_pth_path(full_config, model_idx, epoch, root_path)
        m = model.load_model(full_config, model_idx).load(model_path)
        pred_mask = _inference.run_inference(m, img_np)
        colors = np.array([
            [0, 0, 0],       # Class 0
            [0, 255, 0],     # Class 1
            [128, 0, 128],   # Class 2
        ], dtype=np.uint8)
        colored_mask = _inference.create_colored_mask(pred_mask, colors)
        masks.append(colored_mask)
    return masks


def image_file(path: str) -> Path:
    p = Path(path)

    if not p.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path}")

    if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
        raise argparse.ArgumentTypeError(
            "File must be an image (.jpg, .png, .bmp, .tiff)"
        )

    return p

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Avipe mls inference tool",
        description="Tool to inference with 1+ models after training"
    )
    parser.add_argument('filename', type=image_file, help="Path to the input file for inference")
    parser.add_argument("-c", '--config', help="Path to config.yaml file", required=True)
    parser.add_argument("-m", "--models", help="List of models to inference with separated by ','. Default: all", default="<all>")
    parser.add_argument("-r", "--root-path", help="Root path to experiment weights directory", default=constants.DEFAULT_ROOT_PATH)
    parser.add_argument("-e", "--epoch", type=int, help="Which epoch of the models to run", default="<recent>")
    parser.add_argument("-o", "--output-path", help="path of outputed images", default="output")
    args = parser.parse_args()
    full_config = config.load(args.config)
    model_indexes = []
    epoch_numbers = []
    
    if(args.models == "<all>"):
        for i in range(len(full_config.model)):
            model_indexes.append(i)
    else:
        models = str(args.models).split(",")
        for _model in models:
            found = False
            for idx, model_config in enumerate(full_config.model):
                if model_config.backend.lower() == _model.lower():
                    found = True
                    break
            if not found:
                print(f"Model not found inside config: \"{_model}\"")
                exit(1)
            model_indexes.append(idx)
    
    if(args.epoch == "<recent>"):
        if not Path(args.weights_path).exists():
            print("Didn't find any weights for the models!")
            exit(0)
        for idx in model_indexes:
            found = False
            for epoch in range(full_config.training.epochs,0, -1):
                model_path = utils._get_pth_path(full_config, idx, epoch, args.root_path)
                path = Path(model_path)
                if path.exists() and path.is_file():
                    epoch_numbers.append(idx)
                    found = True
            if not found:
                print(f"Didn't find any weights for {full_config.model[idx].name}")
                exit(0)
    else:
        for idx in model_indexes:
            model_path = utils._get_pth_path(full_config, idx, args.epoch, args.root_path)
            path = Path(model_path)
            if not path.exists() or not path.is_file():
                print(f"Didn't find any weights for epoch {args.epoch} for {full_config.model[idx].name}")
                exit(1)
            epoch_numbers.append(args.epoch)
    
    masks = inference(args.filename, model_indexes, epoch_numbers, full_config, root_path=args.weights_path)
    for idx, mask in enumerate(masks):
        img = Image.fromarray(mask)
        Path(args.output_path).mkdir(exist_ok=True)
        img.save(f"{args.output_path}/{full_config.model[model_indexes[idx]].name}.png")
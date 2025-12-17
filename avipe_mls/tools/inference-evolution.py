import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import imageio.v3 as iio

from .. import constants
from .. import config
from .. import types
from .. import utils
from .. import inference as _inference
from .inference import inference


def inference_evolution(input_file, full_config:types.FullConfig, model_idx: int, epochs:list[int], save_path:str|None = None, output_dir:str=""):
    img = Image.open(input_file).convert("RGB").resize((800,800))
    img_np = np.array(img)
    masks = []
    for epoch in epochs:
        pth_path = utils._get_pth_path(full_config, model_idx, epoch)
        if not Path(pth_path).exists() or not Path(pth_path).is_file():
            print(f"WARNING: skipping {epoch} for model '{full_config.model[model_idx].name}'")
        mask = inference(input_file, [model_idx], [epoch], full_config)[0]
        masks.append(_inference.overlay_mask_on_image(img_np, mask, alpha=0.3))
    
    font_size = 100

    font = ImageFont.load_default(font_size)

    images = []

    for idx, mask in enumerate(masks):
        img = Image.fromarray(mask)
        
        draw = ImageDraw.Draw(img)
        text = f"{idx+1}"
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]

        # Center position
        x = (img.width - text_width) // 2

        # Draw centered text
        draw.text((x, 0), text, fill=(0, 0, 0, 255), font=font)

        # Convert PIL image to NumPy array
        images.append(np.array(img))

    if save_path == None:
        save_path = f"{output_dir}/{full_config.model[model_idx].name}.gif"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(save_path, images, duration=500, loop=1)


def image_file(path: str) -> Path:
    p = Path(path)

    if not p.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path}")

    if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
        raise argparse.ArgumentTypeError(
            "File must be an image (.jpg, .png, .bmp, .tiff)"
        )

    return p

def epochs_type(value):
    if value == "all":
        return "all"
    try:
        return [int(v) for v in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "epochs must be 'all' or a comma-separated list of ints"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Avipe mls inference tool",
        description="Tool to inference with 1+ models after training"
    )
    parser.add_argument('filename', type=image_file, help="Path to the input file for inference")
    parser.add_argument("-c", '--config', help="Path to config.yaml file", required=True)
    parser.add_argument("-m", "--models", help="List of models to inference with separated by ','. Default: all", default="all")
    parser.add_argument("-r", "--root-path", help="Root path to experiment weights directory", default=constants.DEFAULT_ROOT_PATH)
    parser.add_argument("-e", "--epochs", type=epochs_type, help="Which epochs of the models to run separated by ','. Default: all", default="all")
    parser.add_argument("-o", "--output-path", help="path of outputed images", default=None)
    args = parser.parse_args()
    full_config = config.load(args.config)
    
    if(args.output_path == None):
        args.output_path = utils._get_save_path(full_config, args.root_path)

    model_indexes = []

    if(args.models == "all"):
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
    for idx in model_indexes:
        epochs = []
        if(args.epochs == "all"):
            max_epoch = full_config.get_training_for_model(idx).epochs
            epochs = [i for i in range(1, max_epoch+1)]
        else:
            epochs = args.epochs
        inference_evolution(args.filename, full_config, idx, epochs, output_dir=args.output_path)
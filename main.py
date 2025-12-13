import avipe_mls

from PIL import Image
import numpy as np
import torch

if __name__ == "__main__":
    try:
        config = avipe_mls.config.load("example.yaml")
        avipe_mls.trainer.train_model(config)
        dataset = avipe_mls.dataset.create_dataset(config.dataset)
        print("Inference on image:", dataset.images[540])
        img = Image.open(dataset.images[540]).convert("RGB")
        pred_mask = avipe_mls.inference.run_inference(config.model[0], config.dataset, torch.load("grapevista_segmentation-unet_5.pth"), np.array(img))
        colors = np.array([
            [0, 0, 0],       # Class 0
            [0, 255, 0],     # Class 1
            [128, 0, 128],   # Class 2
        ], dtype=np.uint8)
        colored_mask = avipe_mls.inference.create_colored_mask(pred_mask, colors)
        Image.fromarray(colored_mask).save("pred_mask.png")
    except Exception as e:
        print(e)
        exit(-1)
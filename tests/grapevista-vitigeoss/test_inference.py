from PIL import Image
import numpy as np
import torch
from avipe_mls import unet

def predict(model_name, version, image:Image.Image):
        model_manager = unet.models.GetModels().get(model_name)
        model = None
        if model_manager:
            if version and version != "latest":
                model = model_manager.load(version)
            else:
                model = model_manager.load_latest()

        if model_manager and model:
            img, pred_mask = model.predict_image(image)
            img = img.cpu().detach().permute(1, 2, 0).numpy()
            img = Image.fromarray((img * 255).astype(np.uint8))
            if(model_manager.num_classes == 1):
                pred_mask = pred_mask.cpu().detach().squeeze(0)
                pred_mask = torch.sigmoid(pred_mask)
                pred_mask = (pred_mask > 0.5).float().numpy()
                pred_mask = Image.fromarray((pred_mask * 255).astype(np.uint8))
            else:
                pred_mask = pred_mask.cpu().detach()
                pred_mask = torch.softmax(pred_mask, dim=0)          # Convert logits → probabilities
                pred_mask = torch.argmax(pred_mask, dim=0).numpy().astype(np.uint8)   # [H, W] with class indices

                colors = np.array([
                    [0, 0, 0],       # Class 0
                    [0, 255, 0],     # Class 1
                    [128, 0, 128],   # Class 2
                ], dtype=np.uint8)
                height, width = pred_mask.shape
                colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

                for class_idx, color in enumerate(colors):
                    colored_mask[pred_mask == class_idx] = color

                pred_mask = Image.fromarray(colored_mask)
            return img,pred_mask
        print(f"Model \"{model_name}\" not found!")
        return None, None
img, mask = predict("Grapevista - Vitigeoss", "latest", Image.open("/home/rv/Downloads/C3_V4_0700D64C_Aglianico_2022-06-07T09-06-22.jpg"))
img.show()
mask.show("Mask")
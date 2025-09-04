import gradio as gr
from avipe_mls import unet
from PIL import Image
import numpy as np
unet_database = unet.models.GetModels()

def predict(model_name, image:gr.Image):
    model = unet_database.get(model_name)
    if model:
        img, pred_mask = model.predict_image(image)
        img = img.cpu().detach().permute(1, 2, 0).numpy()
        pred_mask = pred_mask.cpu().detach().permute(1, 2, 0).squeeze(-1).numpy()
        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 0] = 1
        img = Image.fromarray((img * 255).astype(np.uint8))
        pred_mask = Image.fromarray((pred_mask * 255).astype(np.uint8))
        return img,pred_mask
    return None, None
def preditor_tab():
    names = [name for name in unet_database.keys()]
    model_chooser = gr.Dropdown(
        choices=names,
        label="Select Model",
        interactive=True
    )
    return gr.Interface(
                fn=lambda model_name, image: predict(model_name, image),
                inputs=[model_chooser, gr.Image(type="pil")],
                outputs=["image","image"],
                flagging_mode="never"
            )
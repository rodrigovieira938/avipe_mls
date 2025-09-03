import gradio as gr
from avipe_mls import unet
from PIL import Image
import numpy as np
unet_database = unet.models.GetModels()

def predict(model_name, image:gr.Image):
    model = unet_database.get(model_name)
    if model:
        _, pred_mask = model.predict_image(image)
        pred_mask = pred_mask.cpu().detach().permute(1, 2, 0).squeeze(-1).numpy()
        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 0] = 1

        img = Image.fromarray((pred_mask * 255).astype(np.uint8))
        return img
    return None

interfaces = []
names = [name for name in unet_database.keys()]
for name, model in unet_database.items():
    interfaces.append(
        gr.Interface(
            fn=lambda image: predict("Carvana", image),
            inputs=[gr.Image(type="pil")],
            outputs=["image"],
            flagging_mode="never"
        )
    )
demo = gr.TabbedInterface(
    interfaces,
    names,
    title="UNet Models"
)
demo.launch(server_name="127.0.0.1", server_port=7860)
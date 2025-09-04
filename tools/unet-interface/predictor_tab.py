import gradio as gr
from avipe_mls import unet
from PIL import Image
import numpy as np
unet_database = unet.models.GetModels()

def predict(model_name, version, image:gr.Image):
    model_manager = unet_database.get(model_name)
    model = None
    if model_manager:
        if version and version != "latest":
            model = model_manager.load(version)
        else:
            model = model_manager.load_latest()

    if model_manager and model:
        img, pred_mask = model.predict_image(image)
        img = img.cpu().detach().permute(1, 2, 0).numpy()
        pred_mask = pred_mask.cpu().detach().permute(1, 2, 0).squeeze(-1).numpy()
        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 0] = 1
        img = Image.fromarray((img * 255).astype(np.uint8))
        pred_mask = Image.fromarray((pred_mask * 255).astype(np.uint8))
        return img,pred_mask
    print(f"Model \"{model_name}\" not found!")
    return None, None
def predictor_tab():
    names = [name for name in unet_database.keys()]
    model_chooser = gr.Dropdown(
        choices=names,
        label="Select Model",
        interactive=True
    )
    def get_versions(model_name):
        #TODO: make this a tuple of (friendly name, filename)
        versions = []
        model_manager = unet_database.get(model_name)
        if model_manager:
            for name in model_manager.checkpoints:
                versions.append((name, name))
            if model_manager.latest:
                versions.append(("latest", "latest"))
        return versions
    versions = get_versions(model_chooser.value)
    version_chooser = gr.Dropdown(
        inputs=[model_chooser],
        #Choose the latest model if it exists, otherwise the first model if it exists else None
        value="latest" if "latest" in [v[0] for v in versions] else (versions[0][0] if versions else None),
        choices=versions,
        label="Select Version",
        interactive=True
    )
    model_chooser.change(
        fn=get_versions,
        inputs=[model_chooser],
        outputs=[version_chooser]
    )
    return gr.Interface(
                fn=predict,
                inputs=[model_chooser,version_chooser, gr.Image(type="pil")],
                outputs=["image","image"],
                flagging_mode="never",
            )
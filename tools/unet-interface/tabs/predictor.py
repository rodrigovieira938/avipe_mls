import gradio as gr
from avipe_mls import unet
from PIL import Image
import numpy as np
from .. import utils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..app import App
import torch

class Predictor:
    def __init__(self, app: 'App') -> None:
        self.app = app
        self.model_chooser = gr.Dropdown(
            choices=[name for name in app.models_db.value.keys()], # Init some default choices
            label="Select Model",
        )
        versions = utils.get_checkpoint_versions(app.models_db.value.get(self.model_chooser.value))
        self.version_chooser = gr.Dropdown(
            inputs=[self.model_chooser],
            #Choose the latest model if it exists, otherwise the first model if it exists else None
            value="latest" if "latest" in [v[0] for v in versions] else (versions[0][0] if versions else None),
            choices=versions,
            label="Select Version",
        )
        # Update the choice of models if the models database changes
        self.app.models_db.change(
            fn=self.update_model_choices,
            inputs=[self.app.models_db],
            outputs=[self.model_chooser]
        )
        # Update the choice of versions if the model changes
        self.model_chooser.change(
            fn=self.update_model_versions_choices,
            inputs=[self.model_chooser],
            outputs=[self.version_chooser]
        )
        self.interface = gr.Interface(
                fn=self.predict,
                inputs=[self.model_chooser, self.version_chooser, gr.Image(type="pil")],
                outputs=["image", "image"],
                flagging_mode="never",
        )
    def update_model_choices(self, models_db):
        return [name for name in models_db.keys()]
    def update_model_versions_choices(self, model_name):
        choices = utils.get_checkpoint_versions(self.app.models_db.value.get(model_name))
        value = None
        if ("latest", "latest") in choices:
            value = "latest"
        return gr.update(choices=choices, value=value)
    def predict(self, model_name, version, image:gr.Image):
        model_manager = self.app.models_db.value.get(model_name)
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

def predictor_tab(app: 'App'):
    return Predictor(app)
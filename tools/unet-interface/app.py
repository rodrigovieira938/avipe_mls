import gradio as gr

from avipe_mls import unet
from .tabs import tabs as tabs_constructors
from .. import utils
import torch

class Sidebar:
    def __init__(self):
        with gr.Sidebar() as sidebar:
            device_dropdown = gr.Dropdown(
                choices=list(utils.query_devices().items()),
                label="Select Device",
                interactive=True
            )
            device_dropdown.change(fn=lambda x: print(torch.device(x)), inputs=device_dropdown, outputs=None)
        self.sidebar = sidebar
        self.device_dropdown = device_dropdown
class App:
    def __init__(self) -> None:
        self.models_db = gr.State(unet.models.GetModels()) # Make this a state so it can used as an input/output
        self.tabs = []
        with gr.Blocks() as main:
            for tab_name, tab_fn in tabs_constructors:
                with gr.Tab(tab_name):
                    tab_fn(self)
                self.tabs.append((tab_name, tab_fn))
            self.sidebar = Sidebar()
        self.main = main
    def run(self):
        self.main.launch()
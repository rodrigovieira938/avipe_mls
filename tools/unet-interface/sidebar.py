import gradio as gr
from .. import utils
import torch

class Sidebar:
    def __init__(self, sidebar, device_dropdown):
        self.sidebar = sidebar
        self.device_dropdown = device_dropdown

def sidebar():
    with gr.Sidebar() as sidebar:
        device_dropdown = gr.Dropdown(
            choices=list(utils.query_devices().items()),
            label="Select Device",
            interactive=True
        )
        device_dropdown.change(fn=lambda x: print(torch.device(x)), inputs=device_dropdown, outputs=None)
    return Sidebar(sidebar, device_dropdown)
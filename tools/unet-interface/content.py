import gradio as gr
from . import predictor_tab
from . import trainer_tab

tabs = [
    ("Predictor", predictor_tab.predictor_tab), 
    ("Trainer", trainer_tab.trainer_tab),  
]

def content():
    with gr.Blocks() as content:
        for tab_name, tab_fn in tabs:
            with gr.Tab(tab_name):
                tab_fn()
    return content
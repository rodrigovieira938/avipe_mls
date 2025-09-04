import gradio as gr
from . import predictor_tab

def _make_interfaces():
    tabbed_interfaces = [predictor_tab.predictor_tab()]
    return (tabbed_interfaces, [str(iface.title) for iface in tabbed_interfaces])

tabs = [
    ("Predictor", predictor_tab.predictor_tab),   
]

def content():
    with gr.Blocks() as content:
        for tab_name, tab_fn in tabs:
            with gr.Tab(tab_name):
                tab_fn()
    return content
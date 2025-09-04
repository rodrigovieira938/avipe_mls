import gradio as gr
from . import predictor_tab

def _make_interfaces():
    tabbed_interfaces = [predictor_tab.predictor_tab()]
    return (tabbed_interfaces, [str(iface.title) for iface in tabbed_interfaces])

def content():
    interfaces, names = _make_interfaces()
    return gr.TabbedInterface(interface_list=interfaces, tab_names=names)
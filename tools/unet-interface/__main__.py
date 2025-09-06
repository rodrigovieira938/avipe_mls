import gradio as gr
from .content import content
from .sidebar import sidebar

if __name__ == "__main__":
    with content() as main:
        sbar = sidebar()

    main.launch()
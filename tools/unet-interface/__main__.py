import gradio as gr
from . import sidebar
from . import content

if __name__ == "__main__":
    with content.content() as main:
        sbar = sidebar.sidebar()

    main.launch()
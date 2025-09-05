import gradio as gr
import pandas as pd
import numpy as np
import random
import plotly.graph_objs as go

from avipe_mls._checkpoint import Checkpoint
from avipe_mls.unet import models

epochs = np.arange(1, 21)
class Insights:
    def __init__(self, dummy_inputs = None) -> None:
        self.data = None
        self.metric_selector = gr.CheckboxGroup(
            choices=["Train Losses", "Validation Losses", "Validation IOU"],
            label="Select Metrics to Plot",
            value=["Train Losses", "Validation Losses", "Validation IOU"]
        )
        self.plot_output = gr.Plot()
        inputs = [self.metric_selector]
        if dummy_inputs is not None:
            inputs.append(dummy_inputs)
        self.metric_selector.change(fn=lambda input: self.plot(input), inputs=inputs, outputs=self.plot_output)
    def set_data(self, checkpoint: Checkpoint | None):
        if(checkpoint is None):
            self.data = None
            return
        self.data = pd.DataFrame({
            "epoch": [i for i in range(0, checkpoint.epoch + 1)],
            "Train Losses": [0] + [i for i in checkpoint.train_losses],
            "Validation IOU": [0] + [i for i in checkpoint.vals_iou],
            "Validation Losses": [0] + [i for i in checkpoint.vals_loss],
        })
    def plot(self, selected_metrics:list[str]):  
        if len(selected_metrics) == 0 or self.data is None:
            return go.Figure()

        fig = go.Figure()
        for metric in selected_metrics:
            fig.add_trace(go.Scatter(
                x=self.data["epoch"],
                y=self.data[metric],
                mode="lines+markers",
                name=metric
            ))
        
        fig.update_layout(
            title="Training Metrics Over Epochs",
            xaxis_title="Epoch",
            yaxis_title="Metric Value",
            legend_title="Metrics",
            template="plotly_white"
        )
        return fig

def insights():
    with gr.Column():
        insights = Insights();
    return insights
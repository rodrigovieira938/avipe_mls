import gradio as gr
from avipe_mls import unet
from ... import utils
from .thread import TrainerThread
from ..common import insights
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...app import App

NOT_TRAINING = 0
TRAINING = 1
STOPPING = 2

class Trainer:
    def __init__(self, app: 'App'):
        self.app = app
        self.state = gr.State(NOT_TRAINING)
        self.trainer_thread = gr.State(None)
        self.poll_timer_state = gr.State(False)
        self.poll_timer = gr.Timer(1)
        with gr.Blocks() as interface:
            with gr.Row():
                self.model_chooser = gr.Dropdown(
                    choices=[name for name in app.models_db.value.keys()], # Init some default choices
                    label="Select Model",
                )
                versions = utils.get_checkpoint_versions(app.models_db.value.get(self.model_chooser.value))
                self.start_checkpoint = gr.Dropdown(
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
                    inputs=[self.model_chooser, self.app.models_db],
                    outputs=[self.start_checkpoint]
                )
            self.state.change(fn=self.on_change_state,
                         inputs=[self.state],
                         outputs=[self.poll_timer_state, self.model_chooser, self.start_checkpoint])
            with gr.Row():
                self.start_training_button = gr.Button("Start Training",variant="primary")
                self.update_status_button = gr.Button("Update Status", variant="secondary")
                self.stop_training_button = gr.Button("Stop Training",variant="stop")
            self.insights = insights.insights()
            self.start_training_button.click(
                fn=self.start_training,
                inputs=[self.model_chooser, self.start_checkpoint, self.state],
                outputs=[self.state, self.trainer_thread]
            )
            self.update_status_button.click(
                fn=lambda *args: self.update_status(*args, manual=True),
                inputs=[self.state, self.trainer_thread],
                outputs=[self.state,self.trainer_thread, self.insights.plot_output]
            )
            self.stop_training_button.click(
                fn=self.stop_training,
                inputs=[self.model_chooser, self.state, self.trainer_thread],
                outputs=[self.state]
            )
            self.poll_timer.tick(
                fn=lambda state, trainer_thread, timer_value: self.update_status(state, trainer_thread, manual=False, enable=timer_value),
                inputs=[self.state, self.trainer_thread, self.poll_timer_state],
                outputs=[self.state, self.trainer_thread, self.insights.plot_output]
            )
    def update_model_choices(self, models_db):
        return [name for name in models_db.keys()]
    def update_model_versions_choices(self, model_name, models_db):
        return utils.get_checkpoint_versions(models_db.get(model_name))
    def on_change_state(self, state_value):
        active = False
        if state_value in (TRAINING, STOPPING):
            active = True
        if state_value == NOT_TRAINING:
            self.model_chooser.interactive = True
            self.start_checkpoint.interactive = True
        else:
            self.model_chooser.interactive = False
            self.start_checkpoint.interactive = False
        return active, self.model_chooser, self.start_checkpoint
    def start_training(self, model_name, version, state):
        if state != NOT_TRAINING:
            print("Cannot start training, already in progress or stopping!")
            return state, None
        print(f"Starting training for model \"{model_name}\" from checkpoint \"{version}\"")
        #Epochs is hight number for testing, should be configurable in the UI
        # TODO: Make epochs configurable
        thread = TrainerThread(model_name, version, epochs=1000)
        thread.run()
        return TRAINING, thread
    def stop_training(self, model_name, state, trainer_thread: TrainerThread):
        if(state == STOPPING):
            print("Already stopping training...")
            return state
        if(state != TRAINING):
            print("Cannot stop training, not currently training!")
            return state
        print(f"Stopping training for model \"{model_name}\"")
        trainer_thread.signal_stop()
        return STOPPING
    def update_status(self, state, trainer_thread:TrainerThread, manual:bool = False, enable:bool = False):
        if not enable and not manual:
            return state, trainer_thread, self.insights.plot_output
        if(manual):
            if state in (NOT_TRAINING):
                print("Cannot update status, not currently training.")
            else:
                print("Manually updating training status...")
        if state == TRAINING or state == STOPPING:
            checkpoints = trainer_thread.get_updates()
            if len(checkpoints) > 0:
                print(f"Received {len(checkpoints)} new checkpoints, latest epoch {checkpoints[-1].epoch}")
                self.insights.set_data(checkpoints[-1])
            if (trainer_thread.is_done()):
                trainer_thread.join()
                if(state == STOPPING):
                    print("Training stopped.")
                else:
                    print("Training completed.")
                return NOT_TRAINING, trainer_thread, self.insights.plot(self.insights.metric_selector.value)
        elif state == STOPPING:
            if(trainer_thread.is_done()):
                trainer_thread.join()
                return NOT_TRAINING, trainer_thread, self.insights.plot(self.insights.metric_selector.value)
        return state, trainer_thread, self.insights.plot(self.insights.metric_selector.value)
def trainer_tab(app: 'App'):
    return Trainer(app)
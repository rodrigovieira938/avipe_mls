import gradio as gr
from avipe_mls import unet
from ... import utils
from .trainer_thread import TrainerThread
from ..common import insights

NOT_TRAINING = 0
TRAINING = 1
STOPPING = 2

def on_change_state(state_value: gr.State, model_chooser: gr.Dropdown, start_checkpoint: gr.Dropdown):
    active = False
    if state_value in (TRAINING, STOPPING):
        active = True
    if state_value == NOT_TRAINING:
        model_chooser.interactive = True
        start_checkpoint.interactive = True
    else:
        model_chooser.interactive = False
        start_checkpoint.interactive = False
    return active, model_chooser, start_checkpoint
def start_training(model_name, version, state):
    if state != NOT_TRAINING:
        print("Cannot start training, already in progress or stopping!")
        return state, None
    print(f"Starting training for model \"{model_name}\" from checkpoint \"{version}\"")
    #Epochs is hight number for testing, should be configurable in the UI
    # TODO: Make epochs configurable
    thread = TrainerThread(model_name, version, epochs=1000)
    thread.run()
    return TRAINING, thread
def update_status(state, insights : insights.Insights, trainer_thread:TrainerThread, manual:bool = False, enable:bool = False):
    if not enable and not manual:
        return state, trainer_thread, insights.plot(insights.metric_selector.value)
    if(manual):
        if state in (NOT_TRAINING, STOPPING):
            print("Cannot update status, not currently training.")
        else:
            print("Manually updating training status...")
    if state == TRAINING or state == STOPPING:
        checkpoints = trainer_thread.get_updates()
        if len(checkpoints) > 0:
            print(f"Received {len(checkpoints)} new checkpoints, latest epoch {checkpoints[-1].epoch}")
            insights.set_data(checkpoints[-1])
        if (trainer_thread.is_done()):
            trainer_thread.join()
            if(state == STOPPING):
                print("Training stopped.")
            else:
                print("Training completed.")
            return NOT_TRAINING, trainer_thread, insights.plot(insights.metric_selector.value)
    elif state == STOPPING:
        if(trainer_thread.is_done()):
            trainer_thread.join()
            return NOT_TRAINING, trainer_thread, insights.plot(insights.metric_selector.value)
    return state, trainer_thread, insights.plot(insights.metric_selector.value)
def stop_training(model_name, state, trainer_thread: TrainerThread):
    if(state != TRAINING):
        print("Cannot stop training, not currently training!")
        return state
    print(f"Stopping training for model \"{model_name}\"")
    trainer_thread.signal_stop()
    return STOPPING

unet_database = unet.models.GetModels()
def trainer_tab():
    names = [name for name in unet_database.keys()]
    state = gr.State(NOT_TRAINING)
    trainer_thread = gr.State(None)
    poll_timer_state = gr.State(False)
    poll_timer = gr.Timer(1)
    with gr.Blocks() as interface:
        with gr.Row():
            model_chooser = gr.Dropdown(
                choices=names,
                label="Select Model",
            )
            versions = utils.get_checkpoint_versions(unet_database.get(model_chooser.value))
            versions.insert(0, ("None", "none")) # Can always start from scratch
            start_checkpoint = gr.Dropdown(
                inputs=[model_chooser],
                value="none",
                choices=versions,
                label="From checkpoint",
            )
            model_chooser.change(
                fn=lambda name: utils.get_checkpoint_versions(unet_database.get(name)),
                inputs=[model_chooser],
                outputs=[start_checkpoint]
            )
            #Need this lambda to capture the model_chooser and start_checkpoint if we used the inputs it would capture the value instead
            state.change(fn=lambda state_value: on_change_state(state_value, model_chooser, start_checkpoint),
                         inputs=[state],
                         outputs=[poll_timer_state, model_chooser, start_checkpoint])
        with gr.Row():
            start_training_button = gr.Button("Start Training",variant="primary")
            update_status_button = gr.Button("Update Status", variant="secondary")
            stop_training_button = gr.Button("Stop Training",variant="stop")
        insights_ = insights.insights()

        start_training_button.click(
            fn=start_training,
            inputs=[model_chooser, start_checkpoint, state],
            outputs=[state, trainer_thread]
        )
        update_status_button.click(
            fn=lambda state, trainer_thread: update_status(state, insights_, trainer_thread, manual=True),
            inputs=[state, trainer_thread],
            outputs=[state,trainer_thread, insights_.plot_output]
        )
        stop_training_button.click(
            fn=stop_training,
            inputs=[model_chooser, state, trainer_thread],
            outputs=[state]
        )
        poll_timer.tick(fn=lambda state, trainer_thread, timer_value: update_status(state, insights_, trainer_thread, manual=False, enable=timer_value), inputs=[state, trainer_thread, poll_timer_state], outputs=[state, trainer_thread, insights_.plot_output])
    return interface
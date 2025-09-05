import gradio as gr
from avipe_mls import unet
from . import utils

NOT_TRAINING = 0
TRAINING = 1

def on_change_state(state_value: gr.State, model_chooser: gr.Dropdown, start_checkpoint: gr.Dropdown):
    active = False
    if state_value == TRAINING:
        active = True
    if state_value == NOT_TRAINING:
        model_chooser.interactive = True
        start_checkpoint.interactive = True
    else:
        model_chooser.interactive = False
        start_checkpoint.interactive = False
    return active, model_chooser, start_checkpoint
def start_training(model_name, version, state):
    if state != 0:
        print("Cannot start training, already in progress or stopping!")
        return state
    print(f"Starting training for model \"{model_name}\" from checkpoint \"{version}\"")
    return TRAINING
def update_status(state, manual:bool = False, enable:bool = False):
    if not enable and not manual:
        return state
    if(manual):
        if state == NOT_TRAINING:
            print("Cannot update status, not currently training.")
        else:
            print("Manually updating training status...")
    return state
def stop_training(model_name, state):
    if(state != TRAINING):
        print("Cannot stop training, not currently training!")
        return state
    print(f"Stopping training for model \"{model_name}\"")
    return NOT_TRAINING

unet_database = unet.models.GetModels()
def trainer_tab():
    names = [name for name in unet_database.keys()]
    state = gr.State(NOT_TRAINING)
    poll_timer_state = gr.State(False)
    poll_timer = gr.Timer(1)
    poll_timer.tick(fn=lambda state, timer_value: update_status(state, manual=False, enable=timer_value), inputs=[state, poll_timer_state], outputs=[state])

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
            start_training_button.click(
                fn=start_training,
                inputs=[model_chooser, start_checkpoint, state],
                outputs=[state]
            )
            update_status_button = gr.Button("Update Status", variant="secondary")
            update_status_button.click(
                fn=lambda *args: update_status(*args, manual=True),
                inputs=[state],
                outputs=[state]
            )
            stop_training_button = gr.Button("Stop Training",variant="stop")
            stop_training_button.click(
                fn=stop_training,
                inputs=[model_chooser, state],
                outputs=[state]
            )
    return interface
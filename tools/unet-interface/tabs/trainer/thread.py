import copy
import threading
import time
from avipe_mls import unet
from avipe_mls._checkpoint import Checkpoint, CheckpointManager
from avipe_mls.unet import models
from avipe_mls.unet._trainer import Trainer
from avipe_mls.unet.trainer_callbacks import CheckpointCallback, CompositeCallback
from ... import utils

import queue
class TrainerThreadCallback(Trainer.Callback):
    def __init__(self, trainer_thread: 'TrainerThread'):
        self.best_iou = None
        self.best_val_loss = None
        self.trainer_thread = trainer_thread
        self.checkpoint_queue = queue.Queue()
    def on_epoch_end(self, trainer, checkpoint: Checkpoint) -> bool:
        print("Epoch done")
        self.checkpoint_queue.put(checkpoint)
        # First draw the UI update then check if we should stop
        if(self.trainer_thread._stop_event.is_set()):
            print("Stopping training...")
            return False # Stop training
        return True  # Continue training
unet_database = unet.models.GetModels()


class BaseEpochCallback(Trainer.Callback):
    def __init__(self, base_checkpoint : Checkpoint, callback: Trainer.Callback, trainer_thread: 'TrainerThread'):
        self.callback = callback
        self.base_checkpoint = base_checkpoint
    def first(self, trainer):
        self.callback.on_epoch_end(trainer, self.base_checkpoint)
    def on_epoch_end(self, trainer, checkpoint: Checkpoint) -> bool:
        checkpoint.epoch += self.base_checkpoint.epoch
        checkpoint.vals_iou = self.base_checkpoint.vals_iou + checkpoint.vals_iou
        checkpoint.vals_loss = self.base_checkpoint.vals_loss + checkpoint.vals_loss
        checkpoint.train_losses = self.base_checkpoint.train_losses + checkpoint.train_losses
        return self.callback.on_epoch_end(trainer, checkpoint)


class TrainerThread:
    def __init__(self, model_name: str, version: str, epochs: int):
        self._model_name = model_name
        self._version = version
        self._epochs = epochs
        self._checkpoint_manager = CheckpointManager("weights")
        self._stop_event = threading.Event()
        self._trainer_callback = TrainerThreadCallback(self)
    def run(self):
        def thread_main():
            model_manager = unet_database.get(self._model_name)
            if not model_manager:
                print(f"Model \"{self._model_name}\" not found!")
                return # This should not happen
            if not self._version:
                self._version = "none"
            start_checkpoint = model_manager.get(self._version)
            if not start_checkpoint:
                print(f"Version \"{self._version}\" of model \"{self._model_name}\" not found!")
                return # This should not happen
            if start_checkpoint is None:
                return # This should not happen
            callbacks = CompositeCallback([
                    CheckpointCallback(self._checkpoint_manager), # First save checkpoints then check for stop and signal updates
                    self._trainer_callback
            ])
            if self._version.upper() != "NONE":
                callbacks = BaseEpochCallback(start_checkpoint, callbacks, self)
            trainer = Trainer(
                model=model_manager.load(self._version),# If model_manager.get(self._version) doesn't return null this doesn't either # type: ignore
                epochs=self._epochs,
                callback=callbacks,
            )
            if self._version.upper() != "NONE":
                callbacks.first(trainer) # type: ignore
            trainer.train(model_manager.dataset)
        self.thread = threading.Thread(target=thread_main)
        self.thread.start()
    def signal_stop(self):
        self._stop_event.set()
    def get_updates(self):
        checkpoints = []
        empty = self._trainer_callback.checkpoint_queue.empty()
        while not self._trainer_callback.checkpoint_queue.empty():
            elem = self._trainer_callback.checkpoint_queue.get()
            if elem != None:
                checkpoints.append(elem)
        return checkpoints
    def is_done(self):
        return not self.thread.is_alive()
    def join(self):
        self.thread.join()
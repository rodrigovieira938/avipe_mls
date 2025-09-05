
from tqdm import tqdm
from .._checkpoint import Checkpoint, CheckpointManager
from ._trainer import Trainer


class CheckpointCallback(Trainer.Callback):
    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
        self.best_iou = None
        self.best_val_loss = None
    def on_epoch_end(self, trainer, checkpoint: Checkpoint) -> bool:
        val_iou = checkpoint.val_iou
        val_loss = checkpoint.val_loss
        save = False
        if val_iou is not None:
            if self.best_iou is None or val_iou > self.best_iou:
                self.best_iou = val_iou
                save = True

        if val_loss is not None:
            if self.best_val_loss is None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save = True

        if save:
            self.checkpoint_manager.save("carvana", checkpoint)
            tqdm.write(f"Model improved at epoch {checkpoint.epoch}, saving checkpoint.")
        return True  # Continue training
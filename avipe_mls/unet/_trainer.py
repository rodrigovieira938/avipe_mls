import uuid
from ._model import UNet
from ._dataset import Dataset
from tqdm import tqdm
import torch as torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
LEARNING_RATE = 3e-4
BATCH_SIZE = 5
EPOCHS = 2

class Trainer:
    class Callback:
        def on_epoch_end(self, trainer: 'Trainer', logs: dict) -> bool:
            """
            Return False to stop training early. If True or None, training continues.
            """
            return True
        def save(self, trainer: 'Trainer', filepath):
            return
    def __init__(self, model: UNet, learning_rate: float = LEARNING_RATE, batch_size: int = BATCH_SIZE, epochs: int = EPOCHS, optimizer = None, criterion = nn.BCEWithLogitsLoss(), callback: Callback | None = None) -> None:
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer if optimizer else optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = criterion
        self.callback = callback
    @staticmethod
    def _compute_iou_components(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
        """Return intersection and union (summed) for IoU."""
        preds = (torch.sigmoid(preds) > threshold).float()
        targets = (targets > 0.5).float()

        intersection = (preds * targets).sum()
        union = ((preds + targets).clamp(0, 1)).sum()

        return intersection.item(), union.item()
    def train(self, dataset: Dataset):
        generator = torch.Generator()
        train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=generator)
        train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=self.batch_size,
                                shuffle=True)
        val_dataloader = DataLoader(dataset=val_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True)
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            train_running_loss = 0
            for idx, img_mask in enumerate(tqdm(train_dataloader)):
                img = img_mask[0].float().to(self.model.device)
                mask = img_mask[1].float().to(self.model.device)

                y_pred = self.model(img)
                self.optimizer.zero_grad()

                loss = self.criterion(y_pred, mask)
                train_running_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()

            train_loss = train_running_loss / (idx + 1)

            self.model.eval()
            val_running_loss = 0
            val_intersection = 0.0
            val_union = 0.0
            with torch.no_grad():
                for idx, img_mask in enumerate(tqdm(val_dataloader)):
                    img = img_mask[0].float().to(self.model.device)
                    mask = img_mask[1].float().to(self.model.device)
                    
                    y_pred = self.model(img)
                    loss = self.criterion(y_pred, mask)

                    val_running_loss += loss.item()
                    inter, union = self._compute_iou_components(y_pred, mask)
                    val_intersection += inter
                    val_union += union

                val_loss = val_running_loss / (idx + 1)
                val_iou = (val_intersection + 1e-6) / (val_union + 1e-6)
            logs = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "epoch": epoch + 1,
                "model": self.model
            }
            self.model.train_loss = train_loss
            self.model.val_loss = val_loss
            self.model.val_iou = val_iou
            self.model.epoch = epoch + 1
            # Generate a new UUID for every epoch
            self.uuid = uuid.uuid4()
            if self.callback:
                if not self.callback.on_epoch_end(self, logs):
                    print("Training stopped early by callback.")
                    break
    def save(self, filepath):
        if self.callback:
            self.callback.save(self, filepath)
        else:
            self.model.save(filepath)
class PrintLogsCallback(Trainer.Callback):
    def on_epoch_end(self, logs: dict) -> bool:
        tqdm.write(f"\n📋 Epoch {logs['epoch']} Logs:")
        for key, value in logs.items():
            if key != "epoch":
                if isinstance(value, float):
                    tqdm.write(f"  {key}: {value:.4f}")
                else:
                    tqdm.write(f"  {key}: {value}")
        return True
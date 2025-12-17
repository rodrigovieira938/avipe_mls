import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from .types import FullConfig
from .model import load_model
from .dataset import create_dataset
from . import constants
from . import utils

def _batch_iou(outputs: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-6):
    """
    outputs: [B, C, H, W] (logits)
    targets: [B, H, W] (long)
    """
    preds = torch.argmax(outputs, dim=1)  # [B, H, W]

    ious = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = targets == cls

        if target_cls.sum() == 0:
            continue  # skip empty class

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        ious.append((intersection + eps) / (union + eps))

    if len(ious) == 0:
        return torch.tensor(0.0, device=outputs.device)

    return torch.mean(torch.stack(ious))

def train_model(config: FullConfig, root_path: str = constants.DEFAULT_ROOT_PATH):
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = create_dataset(config.dataset)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [1.0 - config.training.validation_split, config.training.validation_split])

    batch_size = 4

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    for idx, _ in enumerate(config.model):
        training_config = config.get_training_for_model(idx)
        model = load_model(config, idx)
        model.to(device)
        csv_path = Path(utils._get_csv_path(config, idx, root_path))
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,val_iou\n")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=training_config.lr)

        epochs = training_config.epochs
        for epoch in range(epochs):
            training_loss = 0.0
            model.train()
            train_bar = tqdm(train_loader, desc=f"Training - Epoch {epoch+1}/{epochs}", unit="batch")
            #Training loop
            for batch_idx, (images, masks) in enumerate(train_bar, 1):
                images, masks = images.to(device), masks.to(device)
                if isinstance(criterion, torch.nn.CrossEntropyLoss):
                    masks = masks.squeeze(1).long()

                optimizer.zero_grad()
                outputs = model(images)  # [B, C, H, W]

                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                training_loss += loss.item()
                train_bar.set_postfix({
                    "loss": f"{training_loss/batch_idx:.4f}"
                })
            # Validation loop
            model.eval()
            val_loss = 0.0
            val_iou = 0.0
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f"Validation - Epoch {epoch+1}/{epochs}", unit="batch")
                for batch_idx, (images, masks) in enumerate(val_bar, 1):
                    images, masks = images.to(device), masks.to(device)
                    if isinstance(criterion, torch.nn.CrossEntropyLoss):
                        masks = masks.squeeze(1).long()
                    
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                        iou = _batch_iou(outputs, masks, config.dataset.num_classes)

                        val_loss += loss.item()
                        val_iou += iou.item()
                        val_bar.set_postfix({
                            "loss": f"{val_loss/batch_idx:.4f}",
                            "iou": f"{val_iou / batch_idx:.4f}"
                        })
            avg_train_loss = training_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_iou = val_iou / len(val_loader)
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}"
            )
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{avg_val_iou:.4f}\n")
            #TODO: don't save every epoch
            save_path =  Path(utils._get_pth_path(config, idx, epoch+1))
            save_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(save_path))
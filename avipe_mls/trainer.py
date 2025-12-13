import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from .types import FullConfig
from .model import load_model
from .dataset import create_dataset


def train_model(config: FullConfig, weights_path: str = "./weights"):
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = create_dataset(config.dataset)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [1.0 - config.training.validation_split, config.training.validation_split])

    batch_size = 4

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for idx, _ in enumerate(config.model):
        model = load_model(config, idx)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.training.lr)

        epochs = config.training.epochs
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
                    "train_loss": f"{training_loss/batch_idx:.4f}"
                })
            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f"Validation - Epoch {epoch+1}/{epochs}", unit="batch")
                for batch_idx, (images, masks) in enumerate(val_bar, 1):
                    images, masks = images.to(device), masks.to(device)
                    if isinstance(criterion, torch.nn.CrossEntropyLoss):
                        masks = masks.squeeze(1).long()
                    
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                        val_loss += loss.item()
                        val_bar.set_postfix({
                            "val_loss": f"{val_loss/batch_idx:.4f}"
                        })
            avg_train_loss = training_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            #TODO: don't save every epoch
            Path(weights_path).mkdir(parents=True, exist_ok=True)
            model.save(weights_path, epoch+1)
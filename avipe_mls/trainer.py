import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from .types import FullConfig
from .model import load_model
from .dataset import create_dataset


def train_model(config: FullConfig):
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = create_dataset(config.dataset)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for model_config in config.model:
        model = load_model(model_config, config.dataset)
        model.to(device)
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.training.lr)

        epochs = config.training.epochs
        for epoch in range(epochs):
            epoch_loss = 0.0
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
            for images, masks in train_bar:
                images, masks = images.to(device), masks.to(device)
                if isinstance(criterion, torch.nn.CrossEntropyLoss):
                    masks = masks.squeeze(1).long()

                optimizer.zero_grad()
                outputs = model(images)  # [B, C, H, W]

                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")
            #TODO: don't save every epoch
            torch.save(model.state_dict(), f"{config.name}-{model_config.name}_{epoch+1}.pth")
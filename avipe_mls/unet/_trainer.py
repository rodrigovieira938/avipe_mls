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
    def __init__(self, model: UNet, learning_rate: float = LEARNING_RATE, batch_size: int = BATCH_SIZE, epochs: int = EPOCHS, optimizer = None, criterion = nn.BCEWithLogitsLoss()) -> None:
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer if optimizer else optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = criterion
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
            with torch.no_grad():
                for idx, img_mask in enumerate(tqdm(val_dataloader)):
                    img = img_mask[0].float().to(self.model.device)
                    mask = img_mask[1].float().to(self.model.device)
                    
                    y_pred = self.model(img)
                    loss = self.criterion(y_pred, mask)

                    val_running_loss += loss.item()

                val_loss = val_running_loss / (idx + 1)

            print("-"*30)
            print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
            print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
            print("-"*30)
    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)
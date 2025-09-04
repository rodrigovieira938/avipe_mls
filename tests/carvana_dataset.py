#from avipe_mls.unet import UNet, Trainer, FolderDataset
import matplotlib.pyplot as plt
from PIL import Image
import os
import kagglehub
import torch
from torchvision import transforms
from tqdm import tqdm
from avipe_mls import unet, Checkpoint, CheckpointManager
from pathlib import Path


checkpoint_manager = CheckpointManager("weights")
class CheckpointCallback(unet.Trainer.Callback):
    def __init__(self):
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
            checkpoint_manager.save("carvana", checkpoint)
            tqdm.write(f"Model improved at epoch {checkpoint.epoch}, saving checkpoint.")
        return True  # Continue training

dataset_path = kagglehub.competition_download("carvana-image-masking-challenge")
model = unet.UNet(in_channels=3, num_classes=1)
if os.path.exists("weights/carvana-unet.pth"):
    model.load("weights/carvana-unet.pth")
else:
    print("Model does not exist!")
    print("Starting training")
    if not os.path.exists("weights"):
        os.makedirs("weights")
    training_model = unet.UNet(in_channels=3, num_classes=1)
    trainer = unet.Trainer(training_model, epochs=30, callback=CheckpointCallback())
    transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()])
    dataset = unet.FolderDataset(dataset_path, transform=transform)
    trainer.train(dataset)
    training_model.save("weights/carvana-unet.pth")
    model = training_model
SINGLE_IMG_PATH = f"{dataset_path}/29bb3ece3180_11.jpg"
img = Image.open(SINGLE_IMG_PATH)
img, pred_mask = model.predict_image(img)

pred_mask = pred_mask.cpu().detach().permute(1, 2, 0)
pred_mask[pred_mask < 0]=0
pred_mask[pred_mask > 0]=1

img = img.cpu().detach().permute(1,2,0)

fig = plt.figure()
for i in range(1, 3): 
    fig.add_subplot(1, 2, i)
    if i == 1:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(pred_mask, cmap="gray")
plt.show()
from ._parts import DoubleConv, DownSample, UpSample
import torch as torch
import torch.nn as nn
from torchvision import transforms
import uuid

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)
        
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
        self.to(self.device)

        self.epoch = None
        self.train_loss = None
        self.val_loss = None
        self.val_iou = None
        self.uuid = uuid.uuid4()
    def save(self, filepath):
        checkpoint = {
            'uuid': str(self.uuid),
            'model_state_dict': self.state_dict(),
            "val_loss": self.val_loss,
            "train_loss": self.train_loss,
            "val_iou": self.val_iou,
            "epoch": self.epoch
        }
        torch.save(checkpoint, filepath)
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=torch.device(self.device))
        self.uuid = uuid.UUID(checkpoint['uuid'])
        self.load_state_dict(checkpoint['model_state_dict'])
        self.val_loss = checkpoint["val_loss"]
        self.train_loss = checkpoint["train_loss"]
        self.val_iou = checkpoint["val_iou"]
        self.epoch = checkpoint["epoch"]
    def predict_image(self, image):
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])
        img = transform(image).float().to(self.device)
        img = img.unsqueeze(0)
        with torch.no_grad():
            pred_mask = self(img)
            return img.squeeze(0), pred_mask.squeeze(0)
    def forward(self, x):
       down_1, p1 = self.down_convolution_1(x)
       down_2, p2 = self.down_convolution_2(p1)
       down_3, p3 = self.down_convolution_3(p2)
       down_4, p4 = self.down_convolution_4(p3)

       b = self.bottle_neck(p4)

       up_1 = self.up_convolution_1(b, down_4)
       up_2 = self.up_convolution_2(up_1, down_3)
       up_3 = self.up_convolution_3(up_2, down_2)
       up_4 = self.up_convolution_4(up_3, down_1)

       out = self.out(up_4)
       return out
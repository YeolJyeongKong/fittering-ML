import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torchmetrics

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchsummary import summary
from efficientnet_pytorch import EfficientNet as efficientnet
from torchvision import models

class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.net = efficientnet.from_name('efficientnet-b0')
        self.net._change_in_channels(2)
        self.net._fc = nn.Linear(self.net._fc.in_features, 50)
        self.bn = nn.BatchNorm1d(51)
        self.fc2 = nn.Linear(51, 10)

    def forward(self, image, height):
        x1 = self.net(image)
        x = torch.cat((x1, height.view(-1, 1)), 1)
        x = F.leaky_relu(self.bn(x))
        x = self.fc2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, filters=32, nblocks=5) -> None:
        super(Encoder, self).__init__()
        def CBAP(in_ch, out_ch):
            layers = []
            layers += [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True)]
            layers += [nn.BatchNorm2d(num_features=out_ch)]
            layers += [nn.LeakyReLU()]
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            
            return nn.Sequential(*layers)
        
        blocks = []
        blocks += [CBAP(2, filters)]
        blocks += [CBAP(filters, filters) for _ in range(nblocks-1)]

        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(in_features=8192, out_features=512)
        self.bn = nn.BatchNorm1d(num_features=512)

    def forward(self, x):
        x = self.blocks(x)
        x = x.view(-1, 8192)
        x = self.fc(x)
        x = F.leaky_relu(self.bn(x))
        return x # torch.Size([batch_size, 32, 16, 16])
    

class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        def UCBA(in_ch, out_ch):
            layers = []
            layers += [nn.Upsample(scale_factor=2)]
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)]
            layers += [nn.BatchNorm2d(out_ch)]
            layers += [nn.ReLU()]

            return nn.Sequential(*layers)
        self.fc = nn.Linear(512, 8192)
        self.bn = nn.BatchNorm1d(8192)

        blocks = []
        blocks += [UCBA(32, 32) for _ in range(5)]
        # blocks += [UCBA(32, 2)]

        self.blocks = nn.Sequential(*blocks)

        self.conv2d = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(self.bn(x))
        x = x.view(-1, 32, 16, 16)
        x = self.blocks(x)
        x = F.sigmoid(self.conv2d(x))
        return x




# class AutoEncoder(pl.LightningModule):
#     def __init__(self, filters=32, learning_rate=1e-4):
#         super().__init__()

#         self.save_hyperparameters()
#         self.learning_rate = learning_rate
#         self.filters = 32

#         self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    
#     def forward(self, x):
#         x = self._forward_features(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.log_softmax(self.fc3(x), dim=1)
#         return x
    
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = F.nll_loss(logits, y)

#         preds = torch.argmax(logits, dim=1)
#         acc = self.accuracy(preds, y)
#         self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
#         self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)

#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = F.nll_loss(logits, y)

#         preds = torch.argmax(logits, dim=1)
#         acc = self.accuracy(preds, y)
#         self.log('val_loss', loss, prog_bar=True) # on_step=False, on_epoch=True
#         self.log('val_acc', acc, prog_bar=True) # on_step=False, on_epoch=True
#         return loss

#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = F.nll_loss(logits, y)

#         preds = torch.argmax(logits, dim=1)
#         acc = self.accuracy(preds, y)
#         self.log('test_loss', loss, prog_bar=True) # on_step=False, on_epoch=True
#         self.log('test_acc', acc, prog_bar=True) # on_step=False, on_epoch=True
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
#         return optimizer
    

if __name__ == "__main__":
    device = torch.device('cuda')
    # model = EfficientNet().to(device)
    # image = torch.randn((10, 2, 512, 512)).to(device)
    # height = torch.randn((10, 1)).to(device)
    # sample_output = model(image, height)
    # print(sample_output.shape)

    model = Decoder().to(device)
    summary(model, (512,))
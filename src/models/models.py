import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet as efficientnet
from torchvision import models


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.net = models.resnet18(pretrained=True)
        self.net.fc = nn.Linear(self.net.fc.in_features, 10)

    def forward(self, x):
        return self.net(x)


class EfficientNet_(nn.Module):
    def __init__(self):
        super(EfficientNet_, self).__init__()
        self.net = efficientnet.from_name("efficientnet-b0")
        self.net._change_in_channels(1)
        self.net._fc = nn.Linear(self.net._fc.in_features, 10)

    def forward(self, x):
        x = self.net(x)
        return x


class EfficientNetv2(nn.Module):
    def __init__(self):
        super(EfficientNetv2, self).__init__()
        self.front_net = EfficientNet_()
        self.side_net = EfficientNet_()
        self.fc = nn.Linear(21, 8)

    def forward(self, front, side, height):
        height = height * 0.01
        front = self.front_net(front)
        side = self.side_net(side)
        x = F.leaky_relu(torch.cat((front, side, height.view(-1, 1)), 1))
        x = self.fc(x)
        return x


class SimpleNet_(nn.Module):
    def __init__(self, n_hidden_block, n_feature):
        super(SimpleNet_, self).__init__()
        self.n_feature = n_feature

        def CBADP(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, p=0.5):
            layers = []
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
            ]
            layers += [nn.BatchNorm2d(out_ch)]
            layers += [nn.ReLU()]
            layers += [nn.Dropout(p)]
            layers += [nn.MaxPool2d(2)]

            layer = nn.Sequential(*layers)
            return layer

        block_lst = [CBADP(1, n_feature)]
        block_lst += [CBADP(n_feature, n_feature) for _ in range(n_hidden_block)]
        self.block_lst = nn.Sequential(*block_lst)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_feature, 10)

    def forward(self, x):
        x = self.block_lst(x)
        x = self.avgpool(x)
        x = x.view(-1, self.n_feature)
        x = self.fc(x)
        return x


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.front_net = SimpleNet_(5, 100)
        self.side_net = SimpleNet_(5, 100)
        self.fc = nn.Linear(21, 8)

    def forward(self, front, side, height):
        height = height * 0.01
        front = self.front_net(front)
        side = self.side_net(side)
        x = F.relu(torch.cat((front, side, height.view(-1, 1)), 1))
        x = self.fc(x)
        return x


class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.net = efficientnet.from_name("efficientnet-b0")
        self.net._change_in_channels(2)
        self.net._fc = nn.Linear(self.net._fc.in_features, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc1 = nn.Linear(100, 10)
        self.bn2 = nn.BatchNorm1d(11)
        self.fc2 = nn.Linear(11, 8)

    def forward(self, front, side, height):
        height = height
        image = torch.cat([front, side], dim=1)
        x1 = self.net(image)
        x1 = F.leaky_relu(self.bn1(x1))

        x1 = self.fc1(x1)

        x = torch.cat((x1, height.view(-1, 1)), 1)
        x = F.leaky_relu(self.bn2(x))
        x = self.fc2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, filters=32, nblocks=5) -> None:
        super(Encoder, self).__init__()

        def CBAP(in_ch, out_ch):
            layers = []
            layers += [
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                )
            ]
            layers += [nn.BatchNorm2d(num_features=out_ch)]
            layers += [nn.LeakyReLU()]
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            return nn.Sequential(*layers)

        blocks = []
        blocks += [CBAP(1, filters)]
        blocks += [CBAP(filters, filters) for _ in range(nblocks - 1)]

        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(in_features=8192, out_features=256)

    def forward(self, x):
        x = self.blocks(x)
        x = x.view(-1, 8192)
        x = self.fc(x)
        return x  # torch.Size([-1, 256])


class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()

        def UCBA(in_ch, out_ch):
            layers = []
            layers += [nn.Upsample(scale_factor=2)]
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
            ]
            layers += [nn.BatchNorm2d(out_ch)]
            layers += [nn.ReLU()]

            return nn.Sequential(*layers)

        self.fc = nn.Linear(256, 8192)

        blocks = []
        blocks += [UCBA(32, 32) for _ in range(5)]

        self.blocks = nn.Sequential(*blocks)

        self.conv2d = nn.Conv2d(
            in_channels=32,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = x.view(-1, 32, 16, 16)
        x = self.blocks(x)
        x = F.sigmoid(self.conv2d(x))
        return x  # torch.size([-1, 2, 512, 512])


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


class CombAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontae = AutoEncoder()
        self.sideae = AutoEncoder()

    def forward(self, front, side):
        return self.frontae(front), self.sideae(side)
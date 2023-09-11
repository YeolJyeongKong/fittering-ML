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
    def __init__(self, input_shape, nblocks, filters, latent_dim) -> None:
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
        self.conv_output_shape = input_shape[0] // (2**nblocks)
        self.conv_output_flatten_dim = (self.conv_output_shape) ** 2 * filters
        self.fc = nn.Linear(
            in_features=self.conv_output_flatten_dim, out_features=latent_dim
        )

    def forward(self, x):
        x = self.blocks(x)
        x = x.view(-1, self.conv_output_flatten_dim)
        x = self.fc(x)
        return x  # torch.Size([-1, 256])


class Decoder(nn.Module):
    def __init__(self, input_shape, nblocks, filters, latent_dim) -> None:
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

        self.filters = filters
        self.conv_output_shape = input_shape[0] // (2**nblocks)
        self.conv_output_flatten_dim = (self.conv_output_shape) ** 2 * filters

        self.fc = nn.Linear(latent_dim, self.conv_output_flatten_dim)

        blocks = []
        blocks += [UCBA(filters, filters) for _ in range(nblocks)]

        self.blocks = nn.Sequential(*blocks)

        self.conv2d = nn.Conv2d(
            in_channels=filters,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = x.view(-1, self.filters, self.conv_output_shape, self.conv_output_shape)
        x = self.blocks(x)
        x = F.sigmoid(self.conv2d(x))
        return x  # torch.size([-1, 2, 512, 512])


class AutoEncoder(nn.Module):
    def __init__(self, input_shape, nblocks, filters, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_shape, nblocks, filters, latent_dim)
        self.decoder = Decoder(input_shape, nblocks, filters, latent_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ProductImageEncode(nn.Module):
    def __init__(self):
        super().__init__()
        base = efficientnet.from_name("efficientnet-b0")
        base._blocks = nn.Sequential(*base._blocks)

        layers = list(base.children())[:-6]
        layers += [nn.Conv2d(320, 64, kernel_size=1, stride=1, padding=0, bias=False)]
        layers += [nn.BatchNorm2d(64)]
        layers += [nn.AdaptiveAvgPool2d(1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).flatten(1)


class ProductClassifyBox(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ProductImageEncode()
        self.fc_classify = nn.Linear(64, 50)
        self.fc_box = nn.Linear(64, 4)

    def forward(self, x):
        x = self.encoder(x)
        return self.fc_classify(x), self.fc_box(x)


if __name__ == "__main__":
    from torchinfo import summary

    model = AutoEncoder(input_shape=[512, 512], nblocks=5, filters=32, latent_dim=256)
    summary(model, (16, 1, 512, 512))

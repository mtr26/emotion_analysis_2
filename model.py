import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, groups=1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel_size, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_c, out_c, stride, expand_ratio):
        super().__init__()
        hidden_dim = in_c * expand_ratio
        self.use_residual = (stride == 1 and in_c == out_c)

        layers = []

        # 1×1 expansion
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_c, hidden_dim, kernel_size=1))

        # 3×3 depthwise
        layers.append(
            ConvBNReLU(hidden_dim, hidden_dim,
                       stride=stride, groups=hidden_dim)
        )

        # 1×1 projection (linear)
        layers.append(
            nn.Conv2d(hidden_dim, out_c, kernel_size=1, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_c))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)
        

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=7, input_channels=1):
        super().__init__()

        self.stem = ConvBNReLU(input_channels, 32, stride=2)

        self.blocks = nn.Sequential(
            InvertedResidual(32, 16, 1, 1),

            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6),

            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6),

            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),

            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),

            InvertedResidual(96, 160, 2, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6),

            InvertedResidual(160, 320, 1, 6),
        )

        self.head = nn.Sequential(
            ConvBNReLU(320, 1280, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
    

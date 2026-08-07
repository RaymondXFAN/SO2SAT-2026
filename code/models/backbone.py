# code/models/backbone.py
# MobileNetV3-Small 编码器（~2.5M 参数），So2Sat 32×32 输入 → 256-d 特征
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNetV3Encoder(nn.Module):
    """MobileNetV3-Small backbone: (B, in_ch, 32, 32) → (B, out_dim).
    总参数量 ~2.5M（不含分类头），比 ResNet-18 (~11M) 小 4 倍。
    """
    def __init__(self, in_channels=10, out_dim=256):
        super().__init__()
        # 加载 ImageNet 预训练权重
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # 替换第一层卷积：3通道 → in_channels 通道
        # 原始: Conv2dNormActivation(3, 16, kernel_size=3, stride=2)
        old_conv = self.backbone.features[0][0]  # 第一个 Conv2d
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        self.backbone.features[0][0] = new_conv

        # ★ 替换分类头：576 → out_dim
        # ★★★ 新版 torchvision (≥0.13) MobileNetV3 没有 last_channel 属性！
        # ★★★ 用 classifier[0].in_features 代替（MobileNetV3-Small = 576）
        last_ch = self.backbone.classifier[0].in_features  # 576
        self.backbone.classifier = nn.Sequential(
            nn.Linear(last_ch, out_dim),
        )

        # 初始化新 conv 权重（保持预训练权重分布）
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.backbone(x)   # (B, out_dim)


class ResidualLinear(nn.Module):
    """残差全连接层"""
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)
    def forward(self, x):
        return x + self.fc(x)

# code/models/da.py
# Decoupled Domain Adaptation: S2/S1 对抗对齐，STMN 独立训练
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.w = w
        return x
    @staticmethod
    def backward(ctx, grad):
        return -ctx.w * grad, None


class GradientReversal(nn.Module):
    def __init__(self, w=1.0):
        super().__init__()
        self.w = w
    def forward(self, x):
        return GradientReversalFn.apply(x, self.w)


class DomainDiscriminator(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, 2),  # 源域 vs 目标域
        )
    def forward(self, x):
        return self.net(x)


def decoupled_da_loss(feat_s2, feat_s1, domain_label, adv_w=0.5):
    """仅 S2 和 S1 分支参与域对抗；STMN 分支独立训练不参与。
    domain_label: 0=源域（训练城市）, 1=目标域（测试城市）
    """
    # 合并 S2+S1 特征作为共享特征
    feat_shared = torch.cat([feat_s2, feat_s1], dim=0) if feat_s2.shape == feat_s1.shape else feat_s2
    rev = GradientReversal(adv_w)(feat_shared)
    d_logit = DomainDiscriminator()(rev)
    return F.cross_entropy(d_logit, domain_label)

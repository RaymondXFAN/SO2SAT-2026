# code/models/stmn.py
# Spectral-to-Texture Mapping Network (STMN) — Phase 4: Texture-Specific STMN
# S2(10波段) → Sobel梯度提取 → texture_map(10,32,32) → Student → 256-d → MLP → 256-d
# ★ Phase 4 核心改动: STMN用固定Sobel滤波器预处理S2，提取空间梯度(纹理)信息
# ★ enc_s2提取光谱特征(像素反射率)，STMN提取纹理特征(像素间变化)
# ★ 两者看同一份S2数据但提取根本不同的层面 → 真正互补
# ★ Sobel滤波器固定(不学习) → 保证纹理提取，不退化为冗余光谱编码
# ★ enhance_gate是So2SatNet级别参数，不在STMN内部
import torch
import torch.nn as nn
from .backbone import MobileNetV3Encoder, ResidualLinear


class SpectralStudent(nn.Module):
    """Student 分支：S2 patch (B,10,32,32) → 256-d，用 MobileNetV3-Small 编码。"""
    def __init__(self, out=256, freeze_bn=5):
        super().__init__()
        self.encoder = MobileNetV3Encoder(in_channels=10, out_dim=out)
        self.freeze_bn = freeze_bn
        self._bn_frozen = False

    def forward(self, x):
        return self.encoder(x)  # (B, 256)

    def maybe_freeze_bn(self, epoch):
        """前 freeze_bn 个 epoch 冻结 BN（预训练权重稳定）。"""
        if epoch < self.freeze_bn and not self._bn_frozen:
            for m in self.encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.requires_grad_(False)
            self._bn_frozen = True
        elif epoch >= self.freeze_bn and self._bn_frozen:
            for m in self.encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
            self._bn_frozen = False


class MappingMLP(nn.Module):
    """3 层 MLP: 256 → 512 → 256 → 256, residual + BN + ReLU。"""
    def __init__(self, in_f=256, h=512, out=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f, h), nn.BatchNorm1d(h), nn.ReLU(),
            ResidualLinear(h),
            nn.Linear(h, out), nn.BatchNorm1d(out), nn.ReLU(),
        )

    def forward(self, f):
        return self.net(f)  # (B, 256)


class STMN(nn.Module):
    """STMN: Spectral-to-Texture Mapping Network (Phase 4: Texture-Specific with Sobel).
    S2 → Sobel梯度(固定滤波器) → texture_map → Student(MobileNetV3-Small) → 256-d → MLP → 256-d

    ★ Phase 4 核心改动:
    - STMN用固定Sobel滤波器提取S2的空间梯度(纹理)信息
    - enc_s2看S2原始反射率→光谱特征，STMN看S2梯度→纹理特征
    - 两者提取根本不同的层面，真正互补而非冗余
    - Sobel滤波器固定不学习→保证纹理提取，不退化为冗余光谱编码

    论文叙事: STMN通过空间梯度分析挖掘S2光学数据中的纹理细节，
              与主编码器的光谱特征形成互补，增强城市局部气候区分类

    为什么纹理对LCZ分类有意义:
    - 高层密集建筑 → 强梯度、多方向高频纹理
    - 低层开放建筑 → 稀疏梯度、规则纹理
    - 树木 → 非均匀纹理、自然边界
    - 水体 → 极平滑、几乎无梯度
    - 裸土 → 中等梯度、大面积均匀
    """
    def __init__(self, dim=256, mlp=(512, 256, 256)):
        super().__init__()
        # ── 固定Sobel滤波器：提取空间梯度(纹理) ──
        # depthwise conv: 每个波段独立提取梯度，输出仍是(B,10,32,32)
        self.sobel_x = nn.Conv2d(10, 10, 3, padding=1, bias=False, groups=10)
        self.sobel_y = nn.Conv2d(10, 10, 3, padding=1, bias=False, groups=10)
        # 初始化Sobel核并固定（不学习）
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        for i in range(10):
            self.sobel_x.weight.data[i, 0] = kx
            self.sobel_y.weight.data[i, 0] = ky
        self.sobel_x.requires_grad_(False)
        self.sobel_y.requires_grad_(False)

        # ── Student + MappingMLP ──（结构不变，输入从S2改为texture_map）
        self.student = SpectralStudent(out=dim)
        self.map = MappingMLP(in_f=dim, h=mlp[0], out=dim)

    def forward(self, s2):
        """S2 → Sobel纹理提取 → texture编码 → 256-d纹理增强特征
        输入输出维度与Phase 3完全一致: s2(B,10,32,32) → f_hat(B,256)
        So2SatNet的调用方式不变: f_s2_enhanced = f_s2 + sigmoid(enhance_gate) * f_stmn
        """
        gx = self.sobel_x(s2)           # 水平梯度 (B,10,32,32)
        gy = self.sobel_y(s2)           # 垂直梯度 (B,10,32,32)
        texture_map = torch.sqrt(gx**2 + gy**2 + 1e-6)  # 梯度幅值 (B,10,32,32)
        f_texture = self.student(texture_map)  # texture编码 (B,256)
        f_hat = self.map(f_texture)           # 纹理增强特征 (B,256)
        return f_hat

    def maybe_freeze_bn(self, epoch):
        self.student.maybe_freeze_bn(epoch)

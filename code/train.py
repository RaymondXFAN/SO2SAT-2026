# code/train.py
# So2Sat LCZ42 主训练循环 + 指标 (OA/AA/Kappa/F1) + 早停
# 支持7种实验模式：full, s2_only, s1_only, concat, no_stmn, no_ada, no_gate
# ★ 论文版（路线A）: 支持完整数据集(h5) + --seed参数(多seed均值±std)
# ★ Phase 6: STMN独立第3纹理分支 + 3-分支Ada-MBA + LayerNorm
# ★ 所有模型只返回分类logits，不再有STMN reconstruction loss
# 用法: python code/train.py --mode full --seed 42
import os
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG, DATA, OUT, ROOT, RAW, LCZ_NAMES, FULL_DATA
from datasets import So2SatDataset, make_loader
from models import MobileNetV3Encoder, STMN, AdaMBA


# ── 实验模式 ──
VALID_MODES = ["full", "s2_only", "s1_only", "concat", "no_stmn", "no_ada", "no_gate"]
STMN_MODES = ("full", "no_ada", "no_gate")  # 有STMN的模式（用于BN解冻）
GATE_LOG_MODES = ("full",)  # ★ Phase 6: 只有full有3-分支Ada-MBA+门控(用于gate_stats日志)


# ══════════════════════════════════════════════════════════════
# ── 指标函数（完全保留原代码） ──
# ══════════════════════════════════════════════════════════════

def overall_accuracy(y_true, y_pred):
    return 100.0 * np.mean(y_true == y_pred)


def average_accuracy(y_true, y_pred, n_classes=17):
    """各类别 accuracy 的均值 (AA)"""
    accs = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() > 0:
            accs.append(np.mean(y_pred[mask] == c))
        else:
            accs.append(0.0)
    return 100.0 * np.mean(accs)


def kappa_coefficient(y_true, y_pred, n_classes=17):
    """Cohen's Kappa"""
    cm = np.zeros((n_classes, n_classes))
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    n = cm.sum()
    po = cm.trace() / n
    pe = sum(cm[i].sum() * cm[:, i].sum() for i in range(n_classes)) / (n * n)
    return 100.0 * (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0


def macro_f1(y_true, y_pred, n_classes=17):
    from sklearn.metrics import f1_score
    return 100.0 * f1_score(y_true, y_pred, average='macro',
                             labels=list(range(n_classes)), zero_division=0)


def compute_metrics(y_true, y_pred, n_classes=17):
    oa = overall_accuracy(y_true, y_pred)
    aa = average_accuracy(y_true, y_pred, n_classes)
    kp = kappa_coefficient(y_true, y_pred, n_classes)
    f1 = macro_f1(y_true, y_pred, n_classes)
    return dict(oa=oa, aa=aa, kappa=kp, f1=f1)


# ══════════════════════════════════════════════════════════════
# ── 模型定义：每种实验模式对应一个模型类 ──
# ══════════════════════════════════════════════════════════════

class So2SatNet(nn.Module):
    """完整模型 (full): 3-分支融合 — S2(光谱)+S1(SAR)+STMN(纹理) → LayerNorm → Ada-MBA → head
    ★ Phase 6: STMN不再加法增强S2，而是独立纹理分支！
    ★ Phase 5发现: LayerNorm与加法式增强矛盾 → 归一化摧毁gate的尺度控制
    ★ Phase 6方案: 各分支独立归一化 → 3-分支Ada-MBA注意力融合 → STMN贡献由注意力控制
    ★ 无enhance_gate → STMN贡献完全由Ada-MBA的自注意力+跨注意力+门控决定
    """
    def __init__(self):
        super().__init__()
        self.enc_s2 = MobileNetV3Encoder(in_channels=CFG["s2_bands"], out_dim=256)
        self.enc_s1 = MobileNetV3Encoder(in_channels=CFG["s1_bands"], out_dim=256)
        self.stmn = STMN(dim=256, mlp=(512, 256, 256))
        # ★ Phase 6: 3个LayerNorm，不再有enhance_gate
        self.norm_s2 = nn.LayerNorm(256)   # 归一化S2光谱特征
        self.norm_s1 = nn.LayerNorm(256)    # 归一化S1 SAR特征
        self.norm_stmn = nn.LayerNorm(256)  # 归一化STMN纹理特征
        self.fuse = AdaMBA(d=256, heads=8, n_branches=3)  # ★ 3分支！
        self.head = nn.Linear(256, CFG["n_classes"])

    def forward(self, s2, s1):
        f_s2 = self.enc_s2(s2)
        f_s1 = self.enc_s1(s1)
        f_stmn = self.stmn(s2)  # STMN纹理特征(独立分支，不混进S2)
        # ★ Phase 6: 各分支独立归一化 → 3-分支Ada-MBA
        f_s2_norm = self.norm_s2(f_s2)
        f_s1_norm = self.norm_s1(f_s1)
        f_stmn_norm = self.norm_stmn(f_stmn)
        fused = self.fuse([f_s2_norm, f_s1_norm, f_stmn_norm])
        return self.head(fused)


class S2OnlyModel(nn.Module):
    """对比实验 (s2_only): 只用 S2 光学数据（单模态 baseline）
    enc_s2(10→256) → head(256→17)
    """
    def __init__(self):
        super().__init__()
        self.enc = MobileNetV3Encoder(in_channels=CFG["s2_bands"], out_dim=256)
        self.head = nn.Linear(256, CFG["n_classes"])

    def forward(self, s2):
        f = self.enc(s2)
        return self.head(f)


class S1OnlyModel(nn.Module):
    """对比实验 (s1_only): 只用 S1 SAR 数据（单模态 baseline）
    enc_s1(8→256) → head(256→17)
    """
    def __init__(self):
        super().__init__()
        self.enc = MobileNetV3Encoder(in_channels=CFG["s1_bands"], out_dim=256)
        self.head = nn.Linear(256, CFG["n_classes"])

    def forward(self, s1):
        f = self.enc(s1)
        return self.head(f)


class ConcatModel(nn.Module):
    """对比实验 (concat): S2+S1+STMN → LayerNorm → 早期融合(拼接) baseline
    ★ Phase 6: 3模态拼接(S2+S1+STMN)，与full模型使用相同的输入模态
    enc_s2+enc_s1+stmn → LayerNorm → concat(768) → Linear(768→256) → head(256→17)
    """
    def __init__(self):
        super().__init__()
        self.enc_s2 = MobileNetV3Encoder(in_channels=CFG["s2_bands"], out_dim=256)
        self.enc_s1 = MobileNetV3Encoder(in_channels=CFG["s1_bands"], out_dim=256)
        self.stmn = STMN(dim=256, mlp=(512, 256, 256))
        # ★ Phase 6: 3模态归一化后拼接
        self.norm_s2 = nn.LayerNorm(256)
        self.norm_s1 = nn.LayerNorm(256)
        self.norm_stmn = nn.LayerNorm(256)
        self.fuse = nn.Sequential(nn.Linear(768, 256), nn.ReLU())  # 768 = 256×3
        self.head = nn.Linear(256, CFG["n_classes"])

    def forward(self, s2, s1):
        f_s2 = self.enc_s2(s2)
        f_s1 = self.enc_s1(s1)
        f_stmn = self.stmn(s2)
        f_s2_norm = self.norm_s2(f_s2)
        f_s1_norm = self.norm_s1(f_s1)
        f_stmn_norm = self.norm_stmn(f_stmn)
        fused = self.fuse(torch.cat([f_s2_norm, f_s1_norm, f_stmn_norm], dim=1))
        return self.head(fused)


class NoSTMNModel(nn.Module):
    """消融实验 (no_stmn): 去掉 STMN纹理分支，S2+S1 → LayerNorm → 2-分支Ada-MBA
    ★ Phase 6: 与Phase 5完全一致，2-分支+LayerNorm(不含STMN)
    ★ 证明STMN纹理分支作为第3信息源是有价值的
    """
    def __init__(self):
        super().__init__()
        self.enc_s2 = MobileNetV3Encoder(in_channels=CFG["s2_bands"], out_dim=256)
        self.enc_s1 = MobileNetV3Encoder(in_channels=CFG["s1_bands"], out_dim=256)
        # ★ Phase 5/6: 归一化后进Ada-MBA
        self.norm_s2 = nn.LayerNorm(256)
        self.norm_s1 = nn.LayerNorm(256)
        self.fuse = AdaMBA(d=256, heads=8, n_branches=2)
        self.head = nn.Linear(256, CFG["n_classes"])

    def forward(self, s2, s1):
        f_s2 = self.enc_s2(s2)
        f_s1 = self.enc_s1(s1)
        f_s2_norm = self.norm_s2(f_s2)
        f_s1_norm = self.norm_s1(f_s1)
        fused = self.fuse([f_s2_norm, f_s1_norm])
        return self.head(fused)


class NoAdaModel(nn.Module):
    """消融实验 (no_ada): 3-分支简单平均 — S2+S1+STMN → LayerNorm → (sum)/3 → head
    ★ Phase 6: 3-分支简单平均融合，证明Ada-MBA注意力融合优于简单平均
    ★ 不再使用enhance_gate(STMN是独立分支而非加法增强)
    """
    def __init__(self):
        super().__init__()
        self.enc_s2 = MobileNetV3Encoder(in_channels=CFG["s2_bands"], out_dim=256)
        self.enc_s1 = MobileNetV3Encoder(in_channels=CFG["s1_bands"], out_dim=256)
        self.stmn = STMN(dim=256, mlp=(512, 256, 256))
        # ★ Phase 6: 3个LayerNorm，无enhance_gate
        self.norm_s2 = nn.LayerNorm(256)
        self.norm_s1 = nn.LayerNorm(256)
        self.norm_stmn = nn.LayerNorm(256)
        self.head = nn.Linear(256, CFG["n_classes"])

    def forward(self, s2, s1):
        f_s2 = self.enc_s2(s2)
        f_s1 = self.enc_s1(s1)
        f_stmn = self.stmn(s2)
        f_s2_norm = self.norm_s2(f_s2)
        f_s1_norm = self.norm_s1(f_s1)
        f_stmn_norm = self.norm_stmn(f_stmn)
        fused = (f_s2_norm + f_s1_norm + f_stmn_norm) / 3.0  # 3-分支简单平均
        return self.head(fused)


class NoGateModel(nn.Module):
    """消融实验 (no_gate): 3-分支Ada-MBA(无门控) — cross-attention直接使用，不做α混合
    ★ Phase 6: 证明Ada-MBA内部的门控机制(α)是必要的
    ★ skip_gate=True → 跨注意力结果直接输出，不做 α*ca + (1-α)*F_i 混合
    ★ 不再使用enhance_gate(STMN是独立分支)
    """
    def __init__(self):
        super().__init__()
        self.enc_s2 = MobileNetV3Encoder(in_channels=CFG["s2_bands"], out_dim=256)
        self.enc_s1 = MobileNetV3Encoder(in_channels=CFG["s1_bands"], out_dim=256)
        self.stmn = STMN(dim=256, mlp=(512, 256, 256))
        # ★ Phase 6: 3个LayerNorm，无enhance_gate
        self.norm_s2 = nn.LayerNorm(256)
        self.norm_s1 = nn.LayerNorm(256)
        self.norm_stmn = nn.LayerNorm(256)
        self.fuse = AdaMBA(d=256, heads=8, n_branches=3, skip_gate=True)  # ★ 无门控
        self.head = nn.Linear(256, CFG["n_classes"])

    def forward(self, s2, s1):
        f_s2 = self.enc_s2(s2)
        f_s1 = self.enc_s1(s1)
        f_stmn = self.stmn(s2)
        f_s2_norm = self.norm_s2(f_s2)
        f_s1_norm = self.norm_s1(f_s1)
        f_stmn_norm = self.norm_stmn(f_stmn)
        fused = self.fuse([f_s2_norm, f_s1_norm, f_stmn_norm])
        return self.head(fused)


# ── 模型工厂 ──
def build_model(mode):
    """根据实验模式构建对应模型"""
    model_map = {
        "full":     So2SatNet,
        "s2_only":  S2OnlyModel,
        "s1_only":  S1OnlyModel,
        "concat":   ConcatModel,
        "no_stmn":  NoSTMNModel,
        "no_ada":   NoAdaModel,
        "no_gate":  NoGateModel,
    }
    return model_map[mode]()


# ══════════════════════════════════════════════════════════════
# ── 训练 & 评估（★ Phase 5: 所有模式只有分类loss，LayerNorm在模型内部） ──
# ══════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, opt, sched, crit, device, mode):
    """一个epoch的训练循环。★ Phase 5: 所有模式只有分类loss"""
    model.train()
    total_loss, n = 0.0, 0
    for s2, s1, y, _ in loader:
        y = y.to(device)
        opt.zero_grad()
        if mode == "s2_only":
            logits = model(s2.to(device))
        elif mode == "s1_only":
            logits = model(s1.to(device))
        else:
            # ★ Phase 5: 所有双模态模式统一处理，只有分类loss
            s2, s1 = s2.to(device), s1.to(device)
            logits = model(s2, s1)
        loss = crit(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        opt.step()
        sched.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device, mode):
    """评估函数：根据mode只送需要的输入 ★ Phase 5: 所有模型只返回logits"""
    model.eval()
    Yt, Yp = [], []
    for s2, s1, y, _ in loader:
        if mode == "s2_only":
            logits = model(s2.to(device))
        elif mode == "s1_only":
            logits = model(s1.to(device))
        else:
            logits = model(s2.to(device), s1.to(device))
        Yt += y.tolist()
        Yp += logits.argmax(1).cpu().tolist()
    return compute_metrics(np.array(Yt), np.array(Yp))


@torch.no_grad()
def collect_predictions(model, loader, device, mode):
    """收集测试集所有预测（用于混淆矩阵和逐类OA） ★ Phase 5: 所有模型只返回logits"""
    model.eval()
    all_logits, all_y = [], []
    for s2, s1, y, _ in loader:
        if mode == "s2_only":
            logits = model(s2.to(device))
        elif mode == "s1_only":
            logits = model(s1.to(device))
        else:
            logits = model(s2.to(device), s1.to(device))
        all_logits.append(logits.cpu())
        all_y.append(y)
    return torch.cat(all_logits), torch.cat(all_y)


# ══════════════════════════════════════════════════════════════
# ── 主训练流程 ──
# ══════════════════════════════════════════════════════════════

def main():
    """主训练流程 ★ 论文版: 支持--seed参数 + h5/.pt双模式"""
    # ── 命令行参数 ──
    parser = argparse.ArgumentParser(description="So2Sat LCZ42 实验")
    parser.add_argument("--mode", type=str, default="full",
                        choices=VALID_MODES,
                        help="实验模式: full/s2_only/s1_only/concat/no_stmn/no_ada/no_gate")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（覆盖config默认值42），用于多seed实验")
    args = parser.parse_args()
    mode = args.mode
    seed = args.seed if args.seed is not None else CFG["seed"]

    # ── ★ 输出目录（含seed） ──
    if seed == CFG["seed"]:
        mode_out = OUT / mode             # 默认seed=42: out/full/
    else:
        mode_out = OUT / f"{mode}_seed{seed}"  # 其他seed: out/full_seed123/
    mode_out.mkdir(parents=True, exist_ok=True)

    # ── 打印实验配置 ──
    mode_desc = {
        "full":     "完整模型: S2(光谱)+S1(SAR)+STMN(纹理) → LN → 3-分支Ada-MBA",
        "s2_only":  "对比实验: 只用S2光学数据（单模态baseline）",
        "s1_only":  "对比实验: 只用S1 SAR数据（单模态baseline）",
        "concat":   "对比实验: S2+S1+STMN → LN → 拼接融合（早期融合baseline）",
        "no_stmn":  "消融实验: 去掉STMN，S2+S1 → LN → 2-分支Ada-MBA",
        "no_ada":   "消融实验: S2+S1+STMN → LN → 3-分支简单平均",
        "no_gate":  "消融实验: S2+S1+STMN → LN → 3-分支Ada-MBA(无门控α)",
    }
    print("=" * 60)
    print(f"★ So2Sat LCZ42 训练 — 模式: {mode}")
    print(f"  {mode_desc[mode]}")
    print("=" * 60)
    print(f"  项目根目录: {ROOT}")
    print(f"  数据目录: {DATA}")
    print(f"  输出目录: {mode_out}")
    print(f"  设备: {CFG['device']}")
    print(f"  GPU型号: {torch.cuda.get_device_name(0) if CFG['device']=='cuda' else 'CPU'}")
    print(f"  批量大小: {CFG['batch_size']}")
    print(f"  类别数: {CFG['n_classes']}")
    print(f"  训练轮数: {CFG['epochs']}")
    print(f"  随机种子: {seed}")
    print(f"  数据模式: {'完整数据集(h5)' if FULL_DATA else '子集(.pt)'}")

    device = CFG["device"]

    # ── ★ 设置随机种子 ──
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── 构建模型 ──
    model = build_model(mode).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数: {n_params / 1e6:.2f}M")

    # ── ★ 加载数据（自动选择h5或.pt模式） ──
    if FULL_DATA:
        # ★ 完整数据集: 从h5直接读取
        stats_path = DATA / "stats.json"
        train_h5 = RAW / "training.h5"
        val_h5   = RAW / "validation.h5"
        test_h5  = RAW / "testing.h5"

        if not stats_path.exists():
            print(f"\n✗ {stats_path} 不存在！请先运行 prepare_so2sat.py")
            return
        if not train_h5.exists():
            print(f"\n✗ {train_h5} 不存在！请先下载h5文件")
            return

        tr_loader = make_loader(train_h5, shuffle=True, stats_path=stats_path)
        va_loader = make_loader(val_h5,   shuffle=False, stats_path=stats_path)
        te_loader = make_loader(test_h5,  shuffle=False, stats_path=stats_path)

        print(f"  ★ h5数据加载: train≈320K, val≈82K, test≈82K")
    else:
        # ★ 子集模式: 从.pt文件加载
        train_pt = DATA / "train.pt"
        val_pt   = DATA / "val.pt"
        test_pt  = DATA / "test.pt"

        if not train_pt.exists():
            print(f"\n✗ {train_pt} 不存在！请先运行 prepare_so2sat.py")
            return

        tr_loader = make_loader(str(train_pt), shuffle=True)
        va_loader = make_loader(str(val_pt),   shuffle=False)
        te_loader = make_loader(str(test_pt),  shuffle=False)

        print(f"  ★ .pt数据加载: 子集模式")

    # ── 训练配置 ──
    opt   = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
    crit  = nn.CrossEntropyLoss()

    best_oa, wait = -1, 0
    log_rows = []
    t0 = time.time()

    for ep in range(1, CFG["epochs"] + 1):
        loss = train_one_epoch(model, tr_loader, opt, sched, crit, device, mode)
        vm = evaluate(model, va_loader, device, mode)
        tm = evaluate(model, te_loader, device, mode)

        # ★ Phase 6: Gate值日志改为Ada-MBA gate_stats(不再有enhance_gate)
        # 只对GATE_LOG_MODES(full)计算，需要前向传播获取特征
        gate_info = ""
        mean_gate = None
        if mode in GATE_LOG_MODES and hasattr(model, "fuse") and hasattr(model.fuse, "gate_stats"):
            model.eval()
            with torch.no_grad():
                for s2_b, s1_b, _, _ in va_loader:
                    f_s2 = model.enc_s2(s2_b.to(device))
                    f_s1 = model.enc_s1(s1_b.to(device))
                    f_stmn = model.stmn(s2_b.to(device))
                    feats_norm = [model.norm_s2(f_s2), model.norm_s1(f_s1), model.norm_stmn(f_stmn)]
                    gs = model.fuse.gate_stats(feats_norm)
                    mean_gate = sum(gs.values()) / len(gs)
                    gate_info = f" gates={mean_gate:.3f}"
                    break
            model.train()

        elapsed = time.time() - t0
        print(f"  Epoch {ep:3d}/{CFG['epochs']} | loss={loss:.4f} | "
              f"val OA={vm['oa']:.2f}% AA={vm['aa']:.2f}% Kappa={vm['kappa']:.1f}% F1={vm['f1']:.2f}%{gate_info} | "
              f"test OA={tm['oa']:.2f}% | {elapsed/60:.1f}min")

        log_row = {
            "epoch": ep, "loss": loss,
            "mean_gate": mean_gate,
            "val_oa": vm["oa"], "val_aa": vm["aa"],
            "val_kappa": vm["kappa"], "val_f1": vm["f1"],
            "test_oa": tm["oa"], "test_aa": tm["aa"],
            "test_kappa": tm["kappa"], "test_f1": tm["f1"],
            "elapsed_min": elapsed / 60,
        }
        log_rows.append(log_row)

        # ── 早停 ──
        if vm["oa"] > best_oa:
            best_oa, wait = vm["oa"], 0
            torch.save(model.state_dict(), mode_out / "best_model.pt")
        else:
            wait += 1
            if wait >= CFG["early_stop_patience"]:
                print(f"\n★ 早停触发！最佳 val OA = {best_oa:.2f}%")
                break

        # ── STMN BN 解冻（仅 STMN 模式有 stmn） ──
        if mode in STMN_MODES and hasattr(model, "stmn"):
            model.stmn.maybe_freeze_bn(ep)

    # ── 最终评估 ──
    model.load_state_dict(torch.load(mode_out / "best_model.pt", weights_only=True))
    final = evaluate(model, te_loader, device, mode)
    print(f"\n{'=' * 60}")
    print(f"★ 最终结果 ({mode} mode, best model on test set)")
    print(f"  OA  = {final['oa']:.2f}%")
    print(f"  AA  = {final['aa']:.2f}%")
    print(f"  Kappa = {final['kappa']:.1f}%")
    print(f"  F1  = {final['f1']:.2f}%")
    print(f"{'=' * 60}")

    # ── 收集预测（用于混淆矩阵和逐类 OA） ──
    all_logits, all_y = collect_predictions(model, te_loader, device, mode)
    Yt = all_y.numpy()
    Yp = all_logits.argmax(1).numpy()

    # ── 保存结果 ──
    # 1) 训练日志
    log_path = mode_out / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader()
        w.writerows(log_rows)
    print(f"  训练日志 → {log_path}")

    # 2) 主结果表 (OA/AA/Kappa/F1)
    main_path = mode_out / "table2_main.csv"
    with open(main_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in final.items():
            w.writerow([k, f"{v:.2f}"])
    print(f"  主结果 → {main_path}")

    # 3) 逐类 OA
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    cm = sk_confusion_matrix(Yt, Yp, labels=list(range(17)))

    classwise_path = mode_out / "table5_classwise.csv"
    with open(classwise_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "test_samples", "accuracy"])
        for c in range(17):
            n_samples = cm[c].sum()
            acc = cm[c, c] / n_samples if n_samples > 0 else 0
            w.writerow([c, LCZ_NAMES[c], n_samples, f"{100 * acc:.2f}%"])
    print(f"  逐类OA → {classwise_path}")

    # 5) Ada-MBA Gate值（★ Phase 6: 3-分支Ada-MBA gate_stats，不再有enhance_gate）
    if mode in GATE_LOG_MODES and hasattr(model, "fuse") and hasattr(model.fuse, "gate_stats"):
        model.eval()
        branch_names = ["S2(光谱)", "S1(SAR)", "STMN(纹理)"]
        with torch.no_grad():
            for s2_b, s1_b, _, _ in va_loader:
                f_s2 = model.enc_s2(s2_b.to(device))
                f_s1 = model.enc_s1(s1_b.to(device))
                f_stmn = model.stmn(s2_b.to(device))
                feats_norm = [model.norm_s2(f_s2), model.norm_s1(f_s1), model.norm_stmn(f_stmn)]
                gate_stats = model.fuse.gate_stats(feats_norm)
                break
        gate_path = mode_out / "gate_value.txt"
        with open(gate_path, "w") as f:
            f.write("Ada-MBA gate weights (cross-attention pairs):\n")
            for key, val in gate_stats.items():
                i, j = key.split("-")
                f.write(f"{key} ({branch_names[int(i)]}↔{branch_names[int(j)]}): {val:.4f}\n")
            mean_gate = sum(gate_stats.values()) / len(gate_stats)
            f.write(f"mean_gate: {mean_gate:.4f}\n")
            f.write("interpretation: Ada-MBA cross-attention gate weights for 3-branch fusion\n")
            f.write("  α=0: only original features used (cross-attention ignored)\n")
            f.write("  α=0.5: equal blend of cross-attention and original features\n")
            f.write("  α=1: only cross-attention output used (original features ignored)\n")
        print(f"  Gate值 → {gate_path} (mean_gate={mean_gate:.4f})")

    # 4) 混淆矩阵
    cm_path = mode_out / "confusion_matrix.csv"
    with open(cm_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true"] + LCZ_NAMES)
        for i in range(17):
            w.writerow([LCZ_NAMES[i]] + [cm[i, j] for j in range(17)])
    print(f"  混淆矩阵 → {cm_path}")

    print(f"\n✅ 训练完成！模式={mode}，总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    print(f"  ⚠️ 下载结果后请关机停计费！")


if __name__ == "__main__":
    main()

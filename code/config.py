# code/config.py
# So2Sat LCZ42 实验 — 所有路径 / 超参 / 随机种子集中管理
# ★★★ 关键：所有缓存必须重定向到数据盘 /root/autodl-tmp/.cache/
# ★★★ 系统盘 /root 只有 30G，HF 缓存 7GB+ 直接爆盘！
# ★ 论文版（路线A）：完整数据集 + 2-分支Ada-MBA + LayerNorm
# ★ 提案模型 = no_stmn 配置（S2+S1 → LN → 2-分支Ada-MBA）
# ★ STMN分析在论文Discussion中（Phase 1-6子集实验已提供足够证据）
import os
from pathlib import Path

# ── ★★★ 第一步：缓存重定向（必须在所有 import 之前！） ──
CACHE_ROOT = Path("/root/autodl-tmp/.cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

os.environ["XDG_CACHE_HOME"]   = str(CACHE_ROOT)
os.environ["HF_HOME"]          = str(CACHE_ROOT / "huggingface")
os.environ["HF_DATASETS_CACHE"] = str(CACHE_ROOT / "huggingface/datasets")
os.environ["HF_MODULES_CACHE"]  = str(CACHE_ROOT / "huggingface/modules")
os.environ["PIP_CACHE_DIR"]    = str(CACHE_ROOT / "pip")
os.environ["TORCH_HOME"]       = str(CACHE_ROOT / "torch")
os.environ["HF_ENDPOINT"]      = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# ── 项目根目录 ──（数据盘上）
ROOT = Path(os.environ.get(
    "SO2SAT_ROOT",
    "/root/autodl-tmp/so2sat_exp"
))

# ── 数据路径 ──（全部在数据盘上）
DATA = ROOT / "data"                     # stats.json + .pt 文件（子集模式）
RAW  = DATA / "raw"                      # 原始 h5 文件存这里
OUT  = ROOT / "out"                      # 结果输出
OUT.mkdir(parents=True, exist_ok=True)

# ── ★ 论文版：完整数据集 ──
# 完整数据集使用h5直接读取（不创建巨大.pt文件）
# 预处理只计算stats.json，训练时从h5按需读取+实时归一化
FULL_DATA = True  # ★ 论文版用完整数据集！

# ── 核心超参数 ──
CFG = dict(
    # 硬件
    device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    gpu_batch_size=256,         # ★ 论文版: bs=256（RTX 4090完全够）
    cpu_batch_size=16,

    # 数据
    n_classes=17,
    s2_bands=10,
    s1_bands=8,
    patch=32,
    # ★ FULL_DATA=True时，n_train/n_val/n_test不生效，使用完整数据集
    # ★ 完整数据量: train≈320K, val≈82K, test≈82K
    n_train=30000,              # 子集模式备用值
    n_val=5000,
    n_test=5000,

    # 训练
    epochs=30,                  # ★ 论文版: 30 epochs（数据量大10倍，收敛更快）
    optimizer="AdamW",
    lr=1e-3,
    weight_decay=1e-4,
    scheduler="cosine",
    warmup_epochs=5,
    early_stop_patience=8,      # ★ 30 epochs用8（原来50用10）
    monitor="val_oa",
    grad_clip=1.0,
    seed=42,                    # 默认seed（多seed跑时通过--seed参数覆盖）

    # STMN (★ Phase 6: STMN是独立第3纹理分支，仅full/no_ada/no_gate模式使用)
    stmn=dict(
        dim=256,
        mlp=(512, 256, 256),
        bn_freeze_epochs=5,
    ),

    # Ada-MBA
    ada_mba=dict(
        d=256,
        heads=8,
        n_branches=3,            # full模式用3分支
        # no_stmn模式用2分支（在NoSTMNModel中指定）
    ),

    # Domain Adaptation
    da=dict(adv_weight=0.5, lambda_grid=(0.1, 0.5, 1.0)),
)

# ── 批量大小自动适配 ──
CFG["batch_size"] = CFG["gpu_batch_size"] if CFG["device"] == "cuda" else CFG["cpu_batch_size"]

# ── 论文多seed列表 ──
PAPER_SEEDS = [42, 123, 456]  # no_stmn跑3次取均值±std

# ── LCZ42 类名映射 ──
LCZ_NAMES = [
    "Compact high-rise",       # 0
    "Compact mid-rise",        # 1
    "Compact low-rise",        # 2
    "Open high-rise",          # 3
    "Open mid-rise",           # 4
    "Open low-rise",           # 5
    "Lightweight low-rise",    # 6
    "Large low-rise",          # 7
    "Sparsely built",          # 8
    "Heavy industry",          # 9
    "Dense trees",             # 10
    "Scattered trees",         # 11
    "Bush/scrub",              # 12
    "Low plants",              # 13
    "Bare rock/paved",         # 14
    "Bare soil/sand",          # 15
    "Water",                   # 16
]

# code/utils.py
# 工具函数：计时、日志、路径检查、环境验证
import os
import time
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG, DATA, RAW, ROOT, OUT


def check_env():
    """验证环境配置是否正确"""
    print("=" * 50)
    print("★ 环境验证")
    print("=" * 50)
    issues = []

    # CUDA
    import torch
    if torch.cuda.is_available():
        print(f"  ✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ CUDA 版本: {torch.version.cuda}")
        print(f"  ✓ 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("  ✗ CUDA 不可用！将使用 CPU（非常慢）")
        issues.append("CUDA不可用")

    # 数据盘空间
    disk_info = os.popen("df -h /root/autodl-tmp").read().strip()
    print(f"  数据盘: {disk_info}")

    # 关键库
    libs = ["torch", "numpy", "pandas", "sklearn", "h5py", "datasets"]
    for lib in libs:
        try:
            mod = __import__(lib)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  ✓ {lib}: {ver}")
        except ImportError:
            print(f"  ✗ {lib}: 未安装")
            issues.append(f"{lib}未安装")

    # 数据目录
    print(f"  项目根目录: {ROOT}")
    print(f"  数据目录: {DATA}")
    DATA.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    if issues:
        print(f"\n⚠️ 问题: {', '.join(issues)}")
        return False
    print("\n✅ 环境验证通过！")
    return True


class Timer:
    """计时器"""
    def __init__(self, name=""):
        self.name = name
        self.t0 = time.time()

    def elapsed(self):
        return time.time() - self.t0

    def report(self):
        t = self.elapsed()
        if t > 60:
            print(f"  [{self.name}] {t / 60:.1f} 分钟")
        else:
            print(f"  [{self.name}] {t:.1f} 秒")


def set_seed(seed=42):
    """固定随机种子"""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"  随机种子: {seed}")

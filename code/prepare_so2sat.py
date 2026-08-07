# code/prepare_so2sat.py
# So2Sat LCZ42 数据预处理
# ★ 论文版: 支持两种模式
#   1. 子集模式（FULL_DATA=False）: h5 → 子集采样 → 归一化 → .pt（和之前一样）
#   2. 完整数据集模式（FULL_DATA=True）: h5 → 计算归一化统计量 → stats.json（不创建.pt！）
# ★ 完整数据集训练时从h5按需读取，不需要巨大的.pt文件
import os
import json
import numpy as np
import torch
from pathlib import Path

# ── ★★★ 缓存重定向（必须在 import datasets 之前！） ──
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
CACHE_ROOT = Path("/root/autodl-tmp/.cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"]           = str(CACHE_ROOT / "huggingface")
os.environ["HF_DATASETS_CACHE"]  = str(CACHE_ROOT / "huggingface/datasets")
os.environ["HF_MODULES_CACHE"]  = str(CACHE_ROOT / "huggingface/modules")
os.environ["HF_ENDPOINT"]       = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

from config import CFG, DATA, RAW, ROOT, FULL_DATA


def check_disk_space():
    """检查磁盘空间"""
    print("=" * 60)
    print("★ 磁盘空间检查")
    print("=" * 60)
    tmp_info = os.popen("df -h /root/autodl-tmp").read().strip()
    print(f"  数据盘:\n  {tmp_info}")
    root_info = os.popen("df -h /root").read().strip()
    print(f"  系统盘:\n  {root_info}")
    mem_info = os.popen("free -h").read().strip()
    print(f"\n  内存:\n  {mem_info}")
    return True


def check_memory():
    """检查可用内存"""
    mem_info = os.popen("free -m").read().strip()
    print(f"  内存状态:\n  {mem_info}")
    return True


# ── ★★★ 完整数据集模式：只计算归一化统计量 ──

def compute_stats_from_h5_chunked(h5_path, chunk_size=50000):
    """★ 从h5文件分块计算归一化统计量（mean/std）
    不需要加载全部数据到内存！分块读取，逐块累加统计量。
    内存峰值: ~4GB（每块50K样本），适合任何AutoDL配置。
    """
    import h5py
    print(f"  ★ 分块计算归一化统计量: {h5_path}")
    print(f"    块大小: {chunk_size} 样本/块")

    with h5py.File(str(h5_path), 'r') as f:
        n_total = f['sen2'].shape[0]
        print(f"    总样本数: {n_total}")

        # 累加器（用float64避免精度损失）
        s2_sum = np.zeros(10, dtype=np.float64)
        s2_sq_sum = np.zeros(10, dtype=np.float64)
        s1_sum = np.zeros(8, dtype=np.float64)
        s1_sq_sum = np.zeros(8, dtype=np.float64)
        n_pixels = 0

        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            print(f"    处理块 {start}-{end} ({end/n_total*100:.1f}%)...")

            chunk_s2 = np.array(f['sen2'][start:end], dtype=np.float32)  # (chunk, 32, 32, 10)
            chunk_s1 = np.array(f['sen1'][start:end], dtype=np.float32)  # (chunk, 32, 32, 8)

            # ★ 每个通道的 sum 和 sum_of_squares（跨所有像素）
            s2_sum += chunk_s2.reshape(-1, 10).sum(axis=0)
            s2_sq_sum += (chunk_s2.reshape(-1, 10) ** 2).sum(axis=0)
            s1_sum += chunk_s1.reshape(-1, 8).sum(axis=0)
            s1_sq_sum += (chunk_s1.reshape(-1, 8) ** 2).sum(axis=0)
            n_pixels += (end - start) * 32 * 32

            del chunk_s2, chunk_s1  # 释放块内存

        # 计算mean和std
        s2_mean = (s2_sum / n_pixels).astype(np.float32)
        s2_std = (np.sqrt(s2_sq_sum / n_pixels - (s2_mean.astype(np.float64)) ** 2) + 1e-8).astype(np.float32)
        s1_mean = (s1_sum / n_pixels).astype(np.float32)
        s1_std = (np.sqrt(s1_sq_sum / n_pixels - (s1_mean.astype(np.float64)) ** 2) + 1e-8).astype(np.float32)

    print(f"    ✓ S2 mean: {s2_mean[:3].round(3)}, std: {s2_std[:3].round(3)}")
    print(f"    ✓ S1 mean: {s1_mean[:3].round(3)}, std: {s1_std[:3].round(3)}")
    print(f"    ✓ 总像素数: {n_pixels}")

    return {
        "s2_mean": s2_mean.tolist(),
        "s2_std": s2_std.tolist(),
        "s1_mean": s1_mean.tolist(),
        "s1_std": s1_std.tolist(),
        "n_train": n_total,
        "full_data": True,
    }


def prepare_full_data():
    """★ 完整数据集预处理：只计算stats.json
    训练时从h5文件按需读取，不需要创建.pt文件。
    优点: 预处理只需几分钟，不需要23GB内存，不需要23GB磁盘空间存.pt
    """
    print("=" * 60)
    print("★ ★ 完整数据集模式预处理")
    print("★ 只计算归一化统计量 → stats.json")
    print("★ 训练时从h5按需读取，不需要创建.pt文件！")
    print("=" * 60)

    DATA.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    stats_path = DATA / "stats.json"

    # 检查stats.json是否已存在
    if stats_path.exists():
        with open(stats_path) as f:
            existing = json.load(f)
        if existing.get("full_data", False):
            print(f"\n✅ stats.json 已存在（完整数据集版）！")
            print(f"  n_train={existing['n_train']}")
            print(f"  S2 mean={existing['s2_mean'][:3]}")
            print(f"  不需要重新预处理！")
            return
        else:
            print(f"\n⚠️ stats.json 存在但不是完整数据集版（子集模式）")
            print(f"  需要重新计算完整数据集统计量")

    # 检查h5文件
    train_h5 = RAW / "training.h5"
    val_h5 = RAW / "validation.h5"
    test_h5 = RAW / "testing.h5"

    if not train_h5.exists():
        print(f"\n✗ {train_h5} 不存在！")
        print("  需要先下载h5文件。请参考README中的数据下载说明。")
        print("  或运行: python code/prepare_so2sat.py（会自动尝试HF下载）")
        return

    # 计算训练集归一化统计量（分块，内存安全）
    print(f"\n── Step 1: 计算训练集归一化统计量 ──")
    stats = compute_stats_from_h5_chunked(train_h5, chunk_size=50000)

    # 验证h5文件完整性
    print(f"\n── Step 2: 验证验证集和测试集h5文件 ──")
    import h5py
    for h5_file, name in [(val_h5, "validation"), (test_h5, "testing")]:
        if h5_file.exists():
            with h5py.File(str(h5_file), 'r') as f:
                n = f['sen2'].shape[0]
                print(f"  ✓ {name}: {n} 样本")
                stats[f"n_{name}"] = n
        else:
            print(f"  ✗ {name}: {h5_file} 不存在！")
            return

    # 保存stats.json
    print(f"\n── Step 3: 保存stats.json ──")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ stats.json → {stats_path}")

    print(f"\n✅ 完整数据集预处理完成！")
    print(f"  训练集: ~{stats['n_train']} 样本")
    print(f"  验证集: ~{stats.get('n_validation', '?')} 样本")
    print(f"  测试集: ~{stats.get('n_testing', '?')} 样本")
    print(f"  ★ 训练时将从h5文件按需读取，不需要.pt文件")


# ── ★ 子集模式（原有逻辑，保留作为快速测试备用） ──

def stratified_subset(labels, n, seed=42):
    """分层采样"""
    rng = np.random.RandomState(seed)
    unique_labels = np.unique(labels)
    n_per_class = max(1, n // len(unique_labels))
    idxs = []
    for lbl in unique_labels:
        lbl_idx = np.where(labels == lbl)[0]
        if len(lbl_idx) >= n_per_class:
            chosen = rng.choice(lbl_idx, n_per_class, replace=False)
        else:
            chosen = rng.choice(lbl_idx, len(lbl_idx), replace=False)
        idxs.extend(chosen.tolist())
    if len(idxs) < n:
        remaining = list(set(range(len(labels))) - set(idxs))
        if remaining:
            extra = rng.choice(remaining, n - len(idxs), replace=False)
            idxs.extend(extra.tolist())
    idxs = sorted(idxs[:n])
    return idxs


def load_h5_labels(path):
    """只读labels"""
    import h5py
    print(f"  ★ 只读 labels: {path} ...")
    with h5py.File(str(path), 'r') as f:
        label_raw = np.array(f['label'])
        if label_raw.ndim > 1:
            label = np.argmax(label_raw, axis=1)
        else:
            label = label_raw
    del label_raw
    return label


def load_h5_subset(path, indices):
    """只加载指定索引的S1/S2"""
    import h5py
    indices = np.sort(np.unique(np.asarray(indices)))
    n = len(indices)
    print(f"  ★ 只加载 {n} 个样本: {path} ...")
    with h5py.File(str(path), 'r') as f:
        s1 = np.array(f['sen1'][indices])
        s2 = np.array(f['sen2'][indices])
    print(f"    S1 shape: {s1.shape}, S2 shape: {s2.shape}")
    return s1, s2, indices


def do_normalize_and_save(train_s1, train_s2, train_label,
                           val_s1, val_s2, val_label,
                           test_s1, test_s2, test_label):
    """子集模式: 归一化 + 构建 .pt 文件"""
    DATA.mkdir(parents=True, exist_ok=True)

    s1_mean = train_s1.astype(np.float32).mean(axis=(0, 1, 2))
    s1_std = train_s1.astype(np.float32).std(axis=(0, 1, 2)) + 1e-8
    s2_mean = train_s2.astype(np.float32).mean(axis=(0, 1, 2))
    s2_std = train_s2.astype(np.float32).std(axis=(0, 1, 2)) + 1e-8

    stats = {
        "s2_mean": s2_mean.tolist(), "s2_std": s2_std.tolist(),
        "s1_mean": s1_mean.tolist(), "s1_std": s1_std.tolist(),
        "n_train": len(train_label), "n_val": len(val_label), "n_test": len(test_label),
        "full_data": False,
    }

    def build_samples(s2_all, s1_all, labels_all):
        samples = []
        for i in range(len(labels_all)):
            s2_patch = s2_all[i].astype(np.float32)
            s2_norm = (s2_patch - s2_mean) / s2_std
            s2_ch = np.transpose(s2_norm, (2, 0, 1))

            s1_patch = s1_all[i].astype(np.float32)
            s1_norm = (s1_patch - s1_mean) / s1_std
            s1_ch = np.transpose(s1_norm, (2, 0, 1))

            samples.append({
                "s2": s2_ch, "s1": s1_ch,
                "label": int(labels_all[i]), "city_idx": 0,
            })
        return samples

    train_samples = build_samples(train_s2, train_s1, train_label)
    val_samples = build_samples(val_s2, val_s1, val_label)
    test_samples = build_samples(test_s2, test_s1, test_label)

    torch.save(train_samples, DATA / "train.pt")
    torch.save(val_samples, DATA / "val.pt")
    torch.save(test_samples, DATA / "test.pt")
    json.dump(stats, open(DATA / "stats.json", "w"), indent=2)

    print(f"\n✅ 子集预处理完成！文件在 {DATA}")


def prepare_subset():
    """子集模式预处理（原有逻辑）"""
    print("=" * 60)
    print("★ 子集模式预处理（内存高效版）")
    print("=" * 60)

    train_h5 = RAW / "training.h5"
    val_h5 = RAW / "validation.h5"
    test_h5 = RAW / "testing.h5"

    train_label = load_h5_labels(train_h5)
    val_label = load_h5_labels(val_h5)
    test_label = load_h5_labels(test_h5)

    train_idx = stratified_subset(train_label, CFG["n_train"], seed=CFG["seed"])
    val_idx = stratified_subset(val_label, CFG["n_val"], seed=CFG["seed"] + 1)
    test_idx = stratified_subset(test_label, CFG["n_test"], seed=CFG["seed"] + 2)

    train_s1, train_s2, train_idx_actual = load_h5_subset(train_h5, train_idx)
    val_s1, val_s2, val_idx_actual = load_h5_subset(val_h5, val_idx)
    test_s1, test_s2, test_idx_actual = load_h5_subset(test_h5, test_idx)

    train_label_sub = train_label[train_idx_actual]
    val_label_sub = val_label[val_idx_actual]
    test_label_sub = test_label[test_idx_actual]

    do_normalize_and_save(
        train_s1, train_s2, train_label_sub,
        val_s1, val_s2, val_label_sub,
        test_s1, test_s2, test_label_sub,
    )


def download_hf():
    """HuggingFace 下载（备用）"""
    print("★ HuggingFace 下载（缓存已重定向到数据盘）")
    from datasets import load_dataset
    ds = load_dataset('zhu-xlab/So2Sat-LCZ42')
    print(f"下载完成！训练集 {len(ds['train'])} 样本")
    return ds


def main():
    """主流程：磁盘检查 → 内存检查 → 预处理"""
    print("★ So2Sat LCZ42 数据准备")
    print(f"★ 项目根目录: {ROOT}")
    print(f"★ FULL_DATA={FULL_DATA}")

    check_disk_space()
    check_memory()

    if FULL_DATA:
        # ★ 论文版：完整数据集 → 只算stats.json
        prepare_full_data()
    else:
        # ★ 子集模式 → 算stats + 创建.pt文件
        if (DATA / "train.pt").exists():
            print(f"\n⚠️ train.pt 已存在！如需重新预处理，请先删除:")
            print(f"  rm {DATA}/*.pt {DATA}/stats.json")
            return

        ftp_files = [RAW / "training.h5", RAW / "validation.h5", RAW / "testing.h5"]
        all_exist = all(f.exists() for f in ftp_files)

        if all_exist:
            prepare_subset()
        else:
            missing = [f.name for f in ftp_files if not f.exists()]
            print(f"\n✗ FTP 文件缺失: {missing}")
            print("→ 尝试 HuggingFace 方式...")
            try:
                ds = download_hf()
                # HF下载后自动保存h5到RAW目录
                prepare_subset()
            except Exception as e:
                print(f"\n✗ 下载失败: {e}")
                print("请手动下载h5文件到 " + str(RAW))


if __name__ == "__main__":
    main()

# code/datasets.py
# So2Sat LCZ42 PyTorch Dataset + DataLoader
# ★ 论文版: 支持两种数据加载方式
#   1. .pt文件（子集模式，快速测试）
#   2. h5直接读取（完整数据集，论文正式实验）
# ★ h5方式: 预处理只算stats.json，训练时从h5按需读取+实时归一化
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG, DATA, RAW, FULL_DATA


class So2SatDataset(Dataset):
    """★ 子集模式: 加载预处理好的 .pt 文件（list of dicts格式）
    适用于快速测试，数据量小时使用。
    """
    def __init__(self, pt_path):
        self.samples = torch.load(pt_path, weights_only=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return (
            torch.as_tensor(s["s2"], dtype=torch.float32),
            torch.as_tensor(s["s1"], dtype=torch.float32),
            torch.as_tensor(s["label"], dtype=torch.long),
            int(s.get("city_idx", 0)),
        )


class So2SatDatasetH5(Dataset):
    """★ 完整数据集模式: 直接从h5文件读取 + 实时归一化
    预处理只需计算stats.json（几分钟），不需要创建巨大.pt文件。
    训练时从h5按需读取指定索引的样本，归一化后返回tensor。
    内存占用极小（只有stats + h5文件句柄），适合320K+样本。
    """
    def __init__(self, h5_path, stats_path):
        import h5py
        self.h5_path = str(h5_path)
        self.stats_path = str(stats_path)

        # 加载归一化统计量（极小，<1KB）
        with open(stats_path) as f:
            stats = json.load(f)
        self.s2_mean = np.array(stats["s2_mean"], dtype=np.float32).reshape(1, 1, 10)
        self.s2_std = np.array(stats["s2_std"], dtype=np.float32).reshape(1, 1, 10)
        self.s1_mean = np.array(stats["s1_mean"], dtype=np.float32).reshape(1, 1, 8)
        self.s1_std = np.array(stats["s1_std"], dtype=np.float32).reshape(1, 1, 8)

        # ★ 惰性打开h5文件（支持DataLoader多worker）
        self._h5 = None
        self._labels = None
        self._n = None

    def _open_h5(self):
        """★ 惰性打开：每个DataLoader worker首次访问时打开自己的h5句柄"""
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
            # ★ 一次性读取所有labels（极小，~6MB）→ 避免每次__getitem__都读
            label_raw = np.array(self._h5['label'])
            if label_raw.ndim > 1:
                self._labels = np.argmax(label_raw, axis=1)  # one-hot → int
            else:
                self._labels = label_raw
            self._n = len(self._labels)
            print(f"  ★ So2SatDatasetH5: opened {self.h5_path}, {self._n} samples")

    def __len__(self):
        if self._n is None:
            self._open_h5()
        return self._n

    def __getitem__(self, i):
        self._open_h5()  # 确保h5已打开

        # 从h5读取单样本（按需读取，不需要全部加载）
        s2_raw = np.array(self._h5['sen2'][i], dtype=np.float32)  # (32, 32, 10)
        s1_raw = np.array(self._h5['sen1'][i], dtype=np.float32)  # (32, 32, 8)
        label = int(self._labels[i])

        # 实时归一化
        s2_norm = (s2_raw - self.s2_mean) / self.s2_std
        s1_norm = (s1_raw - self.s1_mean) / self.s1_std

        # transpose: (32,32,C) → (C,32,32)
        s2_ch = np.transpose(s2_norm, (2, 0, 1)).copy()  # .copy()确保内存连续
        s1_ch = np.transpose(s1_norm, (2, 0, 1)).copy()

        return (
            torch.as_tensor(s2_ch, dtype=torch.float32),
            torch.as_tensor(s1_ch, dtype=torch.float32),
            torch.as_tensor(label, dtype=torch.long),
            0,  # city_idx placeholder
        )

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()


def make_loader(data_source, batch_size=None, shuffle=False, stats_path=None):
    """★ 统一创建DataLoader，自动选择.pt或h5模式

    data_source: .pt文件路径（子集模式）或 h5文件路径（完整数据集模式）
    stats_path:  stats.json路径（仅h5模式需要）
    """
    bs = batch_size or CFG["batch_size"]
    num_workers = 4 if CFG["device"] == "cuda" else 0

    if FULL_DATA and stats_path:
        # ★ 完整数据集模式：从h5直接读取
        dataset = So2SatDatasetH5(data_source, stats_path)
    else:
        # ★ 子集模式：从.pt文件加载
        dataset = So2SatDataset(data_source)

    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=CFG["device"] == "cuda",
    )

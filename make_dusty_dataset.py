"""
make_dusty_dataset.py — 离线生成沙尘退化数据集（DAF 受控实验用）
==============================================================

把清晰数据集（如 thermal_pv）按指定沙尘强度，逐张生成退化版本，
保存为新目录，并复制 labels、改写 data.yaml。

为什么用离线方式？
  原 train_yolo_daf.py 的 register_dust_augmentation 只是"准备"了 transform、
  并未真正注册进 Ultralytics 训练管道；且 custom_forward 被移除后 dust_augmenter
  创建了却从未被调用。所以训练图像一直是清晰的，DAF 学不到沙尘信号。
  离线生成退化图像能让沙尘退化 100% 生效、可控、可复现，不依赖改 Ultralytics 内部管道。

用法:
  python daf/make_dusty_dataset.py \
      --src dataset/thermal_pv \
      --dst dataset/thermal_pv_dusty \
      --intensity heavy \
      --seed 42
"""
import os
import sys
import shutil
import argparse
import random

import numpy as np
from PIL import Image
import torch

# 确保能 import 同目录的 dust_simulator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dust_simulator import DustDegradationSimulator

# 沙尘强度预设：clear/light/moderate/heavy 的概率权重
# 越往后的强度越大，退化越明显，给 DAF 留出发挥空间
INTENSITY_PRESETS = {
    'light':    {'clear': 0.30, 'light': 0.35, 'moderate': 0.25, 'heavy': 0.10},
    'moderate': {'clear': 0.15, 'light': 0.25, 'moderate': 0.35, 'heavy': 0.25},
    'heavy':    {'clear': 0.05, 'light': 0.15, 'moderate': 0.30, 'heavy': 0.50},
    'extreme':  {'clear': 0.00, 'light': 0.05, 'moderate': 0.25, 'heavy': 0.70},
}

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def list_images(d):
    if not os.path.isdir(d):
        return []
    return sorted([f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS)])


def process_split(src_img_dir, dst_img_dir, simulator, weights, seed):
    os.makedirs(dst_img_dir, exist_ok=True)
    files = list_images(src_img_dir)
    if not files:
        print(f"  [warn] 无图像: {src_img_dir}")
        return 0
    random.seed(seed)
    cnt = 0
    for f in files:
        # 统一转 RGB（灰度热图会复制成 3 通道，标签不受影响）
        img = Image.open(os.path.join(src_img_dir, f)).convert('RGB')
        arr = np.asarray(img, dtype=np.float32) / 255.0          # (H, W, 3)
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W)
        degraded, _ = simulator.random_degrade(t.unsqueeze(0), level_weights=weights)
        degraded = degraded.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
        out = (degraded * 255.0).round().astype(np.uint8)
        Image.fromarray(out).save(os.path.join(dst_img_dir, f))
        cnt += 1
    print(f"  生成 {cnt} 张 -> {dst_img_dir}")
    return cnt


def rewrite_yaml(src_yaml, dst_yaml, dst_root):
    with open(src_yaml, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    dst_root_abs = os.path.abspath(dst_root)
    out = []
    for line in lines:
        # 只改 path 为输出目录绝对路径，train/val/test 保持相对
        if line.strip().startswith('path:'):
            out.append(f"path: {dst_root_abs}\n")
        else:
            out.append(line)
    with open(dst_yaml, 'w', encoding='utf-8') as f:
        f.writelines(out)


def main():
    ap = argparse.ArgumentParser(description="离线生成沙尘退化数据集")
    ap.add_argument('--src', required=True, help='源数据集根目录(含 images/, labels/, data.yaml)')
    ap.add_argument('--dst', required=True, help='输出数据集根目录')
    ap.add_argument('--intensity', default='heavy',
                    choices=list(INTENSITY_PRESETS.keys()),
                    help='沙尘强度预设')
    ap.add_argument('--seed', type=int, default=42, help='随机种子')
    args = ap.parse_args()

    src_root = os.path.abspath(args.src)
    dst_root = os.path.abspath(args.dst)
    weights = INTENSITY_PRESETS[args.intensity]

    print(f"源: {src_root}")
    print(f"目标: {dst_root}")
    print(f"沙尘强度: {args.intensity} -> 权重 {weights}")

    if not os.path.isdir(os.path.join(src_root, 'images')):
        print(f"[ERROR] 源目录缺少 images/ 子目录: {src_root}")
        return

    simulator = DustDegradationSimulator()

    splits = ['train', 'val', 'test']
    total = 0
    for sp in splits:
        src_img = os.path.join(src_root, 'images', sp)
        dst_img = os.path.join(dst_root, 'images', sp)
        n = process_split(src_img, dst_img, simulator, weights,
                          args.seed + (hash(sp) % 1000))
        total += n
        # 复制 labels（标签与图像通道无关，直接拷贝）
        src_lbl = os.path.join(src_root, 'labels', sp)
        dst_lbl = os.path.join(dst_root, 'labels', sp)
        if os.path.isdir(src_lbl):
            os.makedirs(dst_lbl, exist_ok=True)
            for f in os.listdir(src_lbl):
                if f.lower().endswith('.txt'):
                    shutil.copy2(os.path.join(src_lbl, f), os.path.join(dst_lbl, f))
            print(f"  复制 labels -> {dst_lbl}")

    # 写 data.yaml
    src_yaml = os.path.join(src_root, 'data.yaml')
    dst_yaml = os.path.join(dst_root, 'data.yaml')
    if os.path.isfile(src_yaml):
        rewrite_yaml(src_yaml, dst_yaml, dst_root)
        print(f"data.yaml 已生成: {dst_yaml}")
    else:
        print(f"[warn] 未找到 {src_yaml}，请手动写 data.yaml")

    rel_yaml = os.path.relpath(dst_yaml, os.getcwd())
    print(f"\n✅ 完成！共生成 {total} 张退化图像。")
    print(f"   训练时用: --data {rel_yaml}")


if __name__ == '__main__':
    main()

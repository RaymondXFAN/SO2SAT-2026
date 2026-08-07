# So2Sat LCZ42 论文版实验 — 路线A（完整数据集）
# 数据盘: /root/autodl-tmp (100G)
# ★ 所有数据和结果存储在数据盘，关机后数据不丢失！

## ★ 论文方向

**提案模型**: 2-分支 Ada-MBA + LayerNorm (no_stmn配置)
**核心贡献**: 
1. Ada-MBA融合机制 (+5.46 over concat)
2. LayerNorm对注意力融合的关键性
3. 派生纹理特征的系统失败分析 (6 Phase × 5种STMN设计)

## ★ 数据模式

完整数据集使用**h5直接读取**，预处理只需计算stats.json（几分钟）：
- train: ~320,467 样本
- val: ~81,990 样本  
- test: ~81,990 样本
- 不创建巨大.pt文件（23GB），训练时从h5按需读取+实时归一化

## 快速开始

```bash
# 1. 进入项目目录
cd /root/autodl-tmp/so2sat_exp

# 2. 设置 HF 镜像（每次开新终端都要设置！）
export HF_ENDPOINT=https://hf-mirror.com

# 3. 安装依赖
pip install h5py scikit-learn --quiet

# 4. 预处理（只需几分钟！只算stats.json，不创建.pt）
python code/prepare_so2sat.py

# 5. ★ 论文版一键实验（no_stmn×3seed + 其他模式）
nohup bash run-full.sh > paper_log.txt 2>&1 &
# 监控: tail -f /root/autodl-tmp/so2sat_exp/paper_log.txt
# 预计: 8-10小时

# 6. 也可以单独跑某个模式
python code/train.py --mode no_stmn --seed 42
python code/train.py --mode no_stmn --seed 123  # 多seed
python code/train.py --mode full --seed 42       # STMN有害性验证
```

## 硬件配置

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 4090 (24GB VRAM) |
| CUDA | 12.8 (Driver 570.124.04) |
| PyTorch | cu124 (CUDA 12.4) |
| 数据盘 | /root/autodl-tmp (100G) |
| 系统盘 | /root (30G) |

## 论文实验模式

| 优先级 | 模式 | seed | 理由 |
|--------|------|------|------|
| 🔴必须 | **no_stmn** | 42,123,456 | 提案模型！3次取均值±std |
| 🔴必须 | full | 42 | 证明STMN有害 |
| 🔴必须 | concat | 42 | 证明Ada-MBA>concat |
| 🔴必须 | no_ada | 42 | 证明Ada-MBA>简单平均 |
| 🟡建议 | s2_only | 42 | 单模态baseline |
| 🟡建议 | s1_only | 42 | 单模态baseline |
| 🟡建议 | no_gate | 42 | 证明Ada-MBA门控重要 |

## ⚠️ 重要提醒

1. **完整数据集不需要.pt文件**！只需stats.json + h5文件
2. **h5文件必须存在**：/root/autodl-tmp/so2sat_exp/data/raw/*.h5
3. **跑完必须关机停计费**！
4. **每次开终端重新设置HF镜像**

# Phase 6 部署指南 — STMN独立第3纹理分支 + 3-分支Ada-MBA + LayerNorm

## ★ Phase 6 核心改动总结

| 改动 | Phase 5 (旧) | Phase 6 (新) |
|------|-------------|-------------|
| STMN角色 | 加法增强S2 (`f_s2 + g*f_stmn`) | **独立第3纹理分支** |
| STMN融合方式 | enhance_gate控制加法比例 | **Ada-MBA注意力自适应控制** |
| Ada-MBA分支数 | 2分支(enhanced_S2 + S1) | **3分支(S2 + S1 + STMN)** |
| 归一化 | LayerNorm(enhanced_S2) + LayerNorm(S1) | **3个独立LayerNorm(S2, S1, STMN)** |
| enhance_gate | 有(标量gate) | **去掉！**STMN贡献由Ada-MBA注意力控制 |
| Gate日志 | 单值enhance_gate | **3-分支跨注意力门权重(0-1, 0-2, 1-2)** |

★ 核心公式变化：
- Phase 5: `norm(f_s2 + gate*f_stmn)` → LayerNorm摧毁gate尺度控制 → 归一化悖论
- Phase 6: `[norm_s2(f_s2), norm_s1(f_s1), norm_stmn(f_stmn)]` → 各分支独立归一化 → Ada-MBA自适应融合

★ Phase 5发现：LayerNorm与加法式STMN增强根本矛盾(gate控制被归一化摧毁)
★ Phase 6方案：STMN不再混进S2，而是独立纹理分支 → 无enhance_gate → STMN贡献由Ada-MBA注意力决定

---

## 一、前置条件

### 1. AutoDL实例要求
- GPU: RTX 4090 (24GB VRAM)
- 镜像: PyTorch 2.0 + Python 3.12（AutoDL官方推荐镜像）
- 数据盘: /root/autodl-tmp (100G)
- 系统盘: /root (30G，千万别撑爆！)

### 2. 之前的数据还在吗？
- Phase 1-5预处理好数据应该还在 `/root/autodl-tmp/so2sat_exp/data/` 下
- 检查方法：开机后运行 `ls -lh /root/autodl-tmp/so2sat_exp/data/*.pt`
- 如果 `.pt` 文件都在，**不需要重新预处理**！
- 如果数据丢失（释放实例后数据盘可能清空），需要重新跑预处理

---

## 二、上传代码到云端

### ★ 方法1：WinSCP（推荐，最简单）

将军已经用PuTTY连接云端，最方便的文件上传方式是用WinSCP：

**Step 1: 下载WinSCP**
- 官网：https://winscp.net/eng/download.php
- 安装后打开，新建连接：
  - 协议：SCP
  - 主机名：和PuTTY一样的IP地址
  - 端口：和PuTTY一样的端口
  - 用户名：root
  - 密码：和PuTTY一样的密码
- 连接后直接拖拽文件上传！

**Step 2: 上传代码包**
- 将下载的 `so2sat_exp_phase6.tar.gz` 拖到 `/root/autodl-tmp/` 目录下

### ★ 方法2：scp命令（Windows CMD）

在Windows本地打开CMD（不是PuTTY！），输入：

```bash
scp -P <端口> C:\Users\你的用户名\Desktop\so2sat_exp_phase6.tar.gz root@<IP>:/root/autodl-tmp/
```

例如：
```bash
scp -P 12345 C:\Users\Admin\Desktop\so2sat_exp_phase6.tar.gz root@connect.westb.seetacloud.com:/root/autodl-tmp/
```

> ⚠️ `<端口>` 和 `<IP>` 从AutoDL控制台的SSH连接信息中复制
> ⚠️ Windows路径用反斜杠 `\`，云端路径用正斜杠 `/`
> ⚠️ 如果端口是22可以省略 `-P` 参数

---

## 三、解压代码

### ★ 在PuTTY终端执行

```bash
cd /root/autodl-tmp
mkdir -p /root/autodl-tmp/so2sat_exp
tar -xzf so2sat_exp_phase6.tar.gz -C /root/autodl-tmp/
```

验证：
```bash
ls -la /root/autodl-tmp/so2sat_exp/code/
# 应看到: config.py  datasets.py  train.py  utils.py  prepare_so2sat.py  models/
ls -la /root/autodl-tmp/so2sat_exp/code/models/
# 应看到: __init__.py  backbone.py  ada_mba.py  stmn.py  da.py
ls -la /root/autodl-tmp/so2sat_exp/
# 应看到: README.md  run-all.sh  contribute.md
```

---

## 四、检查之前的数据

```bash
ls -lh /root/autodl-tmp/so2sat_exp/data/*.pt
```

如果看到3个文件（train.pt, val.pt, test.pt），**不用重新预处理**！

如果文件不存在，需要重新预处理：
```bash
cd /root/autodl-tmp/so2sat_exp
export HF_ENDPOINT=https://hf-mirror.com
python code/prepare_so2sat.py
# 等待约30分钟（49GB h5文件读取）
```

---

## 五、安装依赖

```bash
pip install h5py scikit-learn --quiet
```

> ⚠️ PyTorch已在镜像中预装，不需要单独安装
> ⚠️ 缓存已重定向到数据盘，不会爆系统盘

---

## 六、★ Phase 6 实验模式说明

| 模式 | 说明 | 模型结构 |
|------|------|---------|
| **full** | 完整模型 | S2(光谱)+S1(SAR)+STMN(纹理) → LN → 3-分支Ada-MBA → head |
| **s2_only** | S2单模态 | enc_s2 → head（与Phase 5一致） |
| **s1_only** | S1单模态 | enc_s1 → head（与Phase 5一致） |
| **concat** | 早期融合 | S2+S1+STMN → LN → concat(768→256) → head（★ 3模态！） |
| **no_stmn** | 去掉STMN | S2+S1 → LN → 2-分支Ada-MBA → head（★ 与Phase 5一致） |
| **no_ada** | 去掉Ada-MBA | S2+S1+STMN → LN → 3-分支简单平均 → head |
| **no_gate** | 去掉Ada-MBA门控 | S2+S1+STMN → LN → 3-分支Ada-MBA(skip_gate=True) → head |

---

## 七、★ 运行Phase 6实验

### ★★ 推荐方式：一键跑全部7种模式

```bash
cd /root/autodl-tmp/so2sat_exp
nohup bash run-all.sh > phase6_log.txt 2>&1 &
```

**监控训练进度：**
```bash
tail -f /root/autodl-tmp/so2sat_exp/phase6_log.txt
```

**查看是否在运行：**
```bash
ps aux | grep train.py
```

**预计总耗时：3-4小时**（7×30-40分钟）

### ★★ 逐个模式运行（如果想观察每个模式）

```bash
cd /root/autodl-tmp/so2sat_exp

# ★★★ 核心对比（先跑这3个！）
python code/train.py --mode full      # 完整模型
python code/train.py --mode no_stmn   # 去掉STMN
python code/train.py --mode no_gate   # 去掉Ada-MBA门控

# ★★★ 补充对比
python code/train.py --mode no_ada    # 去掉Ada-MBA
python code/train.py --mode s2_only   # S2单模态
python code/train.py --mode s1_only   # S1单模态
python code/train.py --mode concat    # 早期融合
```

### ★ 观察训练过程

每个epoch会输出：
```
  Epoch  1/50 | loss=2.8421 | val OA=15.20% AA=13.56% Kappa=8.5% F1=12.30% gates=0.500 | test OA=14.80% | 0.5min
```

**关键观察点**：
- **gates值**：full模式会输出3-分支Ada-MBA的跨注意力门权重均值
  - gates ≈ 0.5 → Ada-MBA正在学习如何融合3个分支
  - gates > 0.5 → Ada-MBA更依赖跨注意力输出
  - gates < 0.5 → Ada-MBA更依赖原始分支特征
- **OA变化**：
  - 如果 full > no_stmn → STMN作为独立纹理分支有效！🎉🎉🎉
  - 如果 full ≈ no_stmn → Ada-MBA学会了"忽略"STMN
  - 如果 full < no_stmn → STMN仍然有害（不太可能，但需要分析）

---

## 八、★ Phase 6结果收集

### 1. 查看最终结果

每个模式完成后自动输出：
```
★ 最终结果 (full mode, best model on test set)
  OA  = XX.XX%
  AA  = XX.XX%
  Kappa = XX.X%
  F1  = XX.XX%
```

### 2. 一键汇总所有模式结果

```bash
cd /root/autodl-tmp/so2sat_exp
echo "=== Phase 6 结果汇总 ==="
for mode in full s2_only s1_only concat no_stmn no_ada no_gate; do
    if [ -f "out/$mode/table2_main.csv" ]; then
        oa=$(grep "^oa" "out/$mode/table2_main.csv" | cut -d',' -f2)
        aa=$(grep "^aa" "out/$mode/table2_main.csv" | cut -d',' -f2)
        echo "$mode: OA=$oa% AA=$aa%"
    fi
done
```

### 3. 查看Gate值（仅full模式）

```bash
cat /root/autodl-tmp/so2sat_exp/out/full/gate_value.txt
```

输出示例（Phase 6新格式）：
```
Ada-MBA gate weights (cross-attention pairs):
0-1 (S2(光谱)↔S1(SAR)): 0.XXXX
0-2 (S2(光谱)↔STMN(纹理)): 0.XXXX
1-2 (S1(SAR)↔STMN(纹理)): 0.XXXX
mean_gate: 0.XXXX
interpretation: ...
```

★ **关键对比**：
- `0-2`门权重 = S2与STMN纹理的交互强度
- `1-2`门权重 = S1与STMN纹理的交互强度
- 如果STMN纹理分支有用，这两个门权重应该>0.5

### 4. 下载结果到本机

**使用WinSCP**（推荐）：
- 连接后进入 `/root/autodl-tmp/so2sat_exp/out/` 目录
- 将每个模式的文件夹拖到本地

**使用scp命令**（Windows CMD）：
```bash
scp -P <端口> -r root@<IP>:/root/autodl-tmp/so2sat_exp/out/ C:\Users\你的用户名\Desktop\phase6_results\
```

---

## 九、★ 论文关键对比

Phase 6需要验证的核心假设：

| 对比 | Phase 5结果 | Phase 6目标 |
|------|-----------|------------|
| **full vs no_stmn** | 51.10 vs **53.24** (full输❌) | **full > no_stmn** ✅ |
| **full vs concat** | 51.10 vs 47.78 ✅ | full > concat ✅ |
| **full vs no_ada** | 51.10 vs 51.28 ✅ | full > no_ada ✅ |
| **full vs no_gate** | 51.10 vs 49.54 ✅ | full > no_gate ✅ |

★ **最关键的一行**：Phase 6的full必须赢过no_stmn，否则论文不可行！

如果full > no_stmn → STMN独立纹理分支+3-分支Ada-MBA有效 → 可以写论文！🎉
如果full ≤ no_stmn → STMN仍然有害 → 需要Phase 7或放弃STMN路线

---

## 十、⚠️ 常见问题 FAQ

### Q1: 数据预处理报错 "h5文件不存在"
**A**: 需要先下载h5文件或使用HuggingFace下载。设置HF镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
python code/prepare_so2sat.py
```

### Q2: `(venv)(base)` 双环境叠加
**A**: 这是AutoDL镜像的默认状态，不影响运行。如果想去掉：
```bash
conda deactivate   # 先退出venv
conda deactivate   # 再退出base
# 或者直接忽略，不影响训练
```

### Q3: CUDA out of memory
**A**: batch_size=64在RTX 4090上应该没问题。如果报错，在config.py中改：
```python
gpu_batch_size=32,  # 从64改为32
```

### Q4: 训练速度太慢
**A**: 每个epoch约30秒，50 epochs约25分钟。如果太慢，检查：
```bash
nvidia-smi  # 确认GPU被使用
python -c "import torch; print(torch.cuda.is_available())"  # 确认CUDA可用
```

### Q5: 结果波动太大
**A**: CUDA操作有非确定性，同一模型不同运行可能差1-2%。这是正常的。
所有Phase内部对比仍然有效（同批次训练）。

### Q6: 想中途停止训练
**A**: 在PuTTY终端按 `Ctrl+C`。如果用nohup后台运行：
```bash
ps aux | grep train.py  # 找到进程ID
kill <PID>              # 停止进程
```

---

## 十一、关机停计费

**★ 跑完必须关机！AutoDL按小时计费！**

1. 先下载结果到本地（WinSCP或scp）
2. 然后在AutoDL控制台点击"关机"
3. 如果只是暂停（数据保留），选"关机（保留数据）"
4. 如果不再需要，选"释放实例"（数据全部删除！）

# Phase 3 部署指南 — STMN改为S2纹理增强器 + 2分支Ada-MBA

## ★ Phase 3 核心改动总结

| 改动 | Phase 2 (旧) | Phase 3 (新) |
|------|-------------|-------------|
| STMN角色 | Ada-MBA第3独立分支 | S2纹理增强器 |
| STMN输出 | gated pseudo-S1 | 完整f_hat(不gated) |
| STMN loss | reconstruction loss ||F_hat-F_s1||² | **去掉！**只用分类loss反传 |
| Gate位置 | STMN内部(gate_alpha) | So2SatNet级别(enhance_gate) |
| Gate初始值 | sigmoid(-2)=0.12 | sigmoid(-1)=0.27 |
| Ada-MBA分支 | 3分支(S2+S1+STMN) | **2分支(enhanced_S2+S1)** |
| 模型forward返回 | (logits, f_hat, f_s1) for STMN modes | **只返回logits** |

★ 关键公式：`f_s2_enhanced = f_s2 + sigmoid(enhance_gate) * f_stmn`
- gate=0 → f_s2_enhanced = f_s2 (等同no_stmn，安全兜底)
- gate>0 → STMN增强S2纹理特征
- STMN不再模仿S1，完全通过分类loss学习有用映射

---

## 一、前置条件

### 1. AutoDL实例要求
- GPU: RTX 4090 D (24GB)
- 镜像: PyTorch 2.0 + Python 3.12
- 数据盘: /root/autodl-tmp (100G)
- 系统盘: /root (30G，千万别撑爆！)

### 2. 之前的数据还在吗？
- Phase 1/2预处理好数据应该还在 `/root/autodl-tmp/so2sat_exp/data/` 下
- 检查方法：开机后运行 `ls -lh /root/autodl-tmp/so2sat_exp/data/*.pt`
- 如果 `.pt` 文件都在，**不需要重新预处理**！
- 如果数据丢失（释放实例后数据盘可能清空），需要重新跑预处理

---

## 二、上传代码到云端

### ★ 使用PuTTY + WinSCP（推荐）

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
- 将下载的 `so2sat_exp.tar.gz` 拖到 `/root/autodl-tmp/` 目录下

### ★ 使用scp命令（PuTTY终端）

在Windows本地打开CMD（不是PuTTY！），输入：

```bash
scp -P <端口> C:\Users\你的用户名\Desktop\so2sat_exp.tar.gz root@<IP>:/root/autodl-tmp/
```

例如：
```bash
scp -P 12345 C:\Users\Admin\Desktop\so2sat_exp.tar.gz root@connect.westb.seetacloud.com:/root/autodl-tmp/
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
tar -xzf so2sat_exp.tar.gz -C /root/autodl-tmp/
```

验证：
```bash
ls -la /root/autodl-tmp/so2sat_exp/code/
# 应看到: config.py  datasets.py  train.py  models/
ls -la /root/autodl-tmp/so2sat_exp/code/models/
# 应看到: __init__.py  backbone.py  ada_mba.py  stmn.py  da.py
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
python code/datasets.py
# 等待约30分钟（49GB h5文件读取）
```

---

## 五、安装依赖

```bash
cd /root/autodl-tmp/so2sat_exp
python -m pip install h5py scikit-learn --quiet
```

> ⚠️ PyTorch已在镜像中预装，不需要单独安装
> ⚠️ 缓存已重定向到数据盘，不会爆系统盘

---

## 六、★ Phase 3 实验模式说明

| 模式 | 说明 | 模型结构 |
|------|------|---------|
| **full** | 完整模型 | STMN增强S2 + Ada-MBA(2分支) + enhance_gate |
| **s2_only** | S2单模态 | enc_s2 → head |
| **s1_only** | S1单模态 | enc_s1 → head |
| **concat** | 早期融合 | enc_s2 + enc_s1 → concat(512→256) → head |
| **no_stmn** | 去掉STMN | S2+S1 + Ada-MBA(2分支) → head |
| **no_ada** | 去掉Ada-MBA | STMN增强S2 + S1简单平均 → head |
| **no_gate** | 去掉Gate | STMN完整增强S2(gate固定1.0) + Ada-MBA(2分支) → head |

---

## 七、★ 运行Phase 3实验（逐个模式）

### ★★ 推荐顺序（按重要性排列）

**核心对比（必须跑）：**
1. `full` — 完整模型（STMN增强S2 + Ada-MBA + Gate）
2. `no_stmn` — 去掉STMN（与Phase 2对照）
3. `no_gate` — 去掉Gate（消融Gate必要性）

**补充对比（建议跑）：**
4. `s2_only` — S2单模态baseline
5. `s1_only` — S1单模态baseline
6. `concat` — 早期融合baseline
7. `no_ada` — 去掉Ada-MBA

### ★ 执行命令

每个模式约30-40分钟（50 epoch，30000训练样本），逐个运行：

```bash
cd /root/autodl-tmp/so2sat_exp

# ★★★ 核心3个模式（先跑这3个！）
python code/train.py --mode full
python code/train.py --mode no_stmn
python code/train.py --mode no_gate

# 补充4个模式
python code/train.py --mode s2_only
python code/train.py --mode s1_only
python code/train.py --mode concat
python code/train.py --mode no_ada
```

### ★ 观察训练过程

每个epoch会输出：
```
  Epoch  1/50 | loss=2.8421 | val OA=15.20% AA=13.56% Kappa=8.5% F1=12.30% gate=0.2694 | test OA=14.80% | 0.5min
```

**关键观察点**：
- **gate值变化**：full模式会输出 `gate=X.XXXX`，观察gate从0.27如何变化
  - 如果gate逐渐增大 → STMN学到有用的纹理特征！🎉
  - 如果gate逐渐减小 → STMN仍然有害（但不会崩溃，因为有安全兜底）
  - 如果gate保持稳定 → STMN贡献稳定

- **OA变化**：与Phase 2对照
  - Phase 2 full OA=50.38% → Phase 3 full应该≥50.38%（至少不差）
  - 如果full > no_stmn → STMN增强有效！🎉🎉🎉
  - 如果full ≈ no_stmn → Gate关闭了STMN（安全兜底，不崩溃）
  - 如果full < no_stmn → 新架构有问题（不太可能）

---

## 八、★ Phase 3结果收集

### 1. 查看最终结果

每个模式完成后自动输出：
```
★ 最终结果 (full mode, best model on test set)
  OA  = XX.XX%
  AA  = XX.XX%
  Kappa = XX.X%
  F1  = XX.XX%
```

### 2. 查看Gate值（仅full/no_ada/no_gate模式）

```bash
cat /root/autodl-tmp/so2sat_exp/out/full/gate_value.txt
cat /root/autodl-tmp/so2sat_exp/out/no_ada/gate_value.txt
cat /root/autodl-tmp/so2sat_exp/out/no_gate/gate_value.txt
```

输出示例：
```
enhance_gate (raw): -0.5234
enhance_gate (sigmoid): 0.3710
interpretation: STMN contributes 37.1% of its features to enhance S2
```

### 3. 查看训练日志

```bash
cat /root/autodl-tmp/so2sat_exp/out/full/train_log.csv
# 每行包含: epoch, loss, gate_value, val_oa, val_aa, val_kappa, val_f1, test_oa, ...
```

---

## 九、★ Phase 3 vs Phase 2 对照预期

| 模式 | Phase 2结果 | Phase 3预期 | 预期原因 |
|------|------------|------------|---------|
| full | 50.38% | ≥50.38% | gate关闭时等同no_stmn(51.62%)，安全兜底 |
| no_stmn | 51.62% | ~51.62% | 结构不变 |
| no_gate | 50.60% | ？ | gate固定1.0，STMN可能干扰或增强 |
| s2_only | 49.54% | ~49.54% | 结构不变 |
| s1_only | 24.22% | ~24.22% | 结构不变 |

**★ 关键判断标准**：
- **full > no_stmn** → STMN增强有效！论文核心贡献确立！
- **full ≈ no_stmn** → Gate关闭了STMN，但不崩溃（安全兜底有效）
- **full < no_stmn** → Phase 3设计有问题（不太可能，因为有安全兜底）

---

## 十、下载结果

### 使用WinSCP下载

连接WinSCP后，进入 `/root/autodl-tmp/so2sat_exp/out/` 目录，拖拽每个模式的文件夹到本地。

### 使用scp下载（CMD）

```bash
scp -P <端口> -r root@<IP>:/root/autodl-tmp/so2sat_exp/out/ C:\Users\你的用户名\Desktop\phase3_results\
```

---

## 十一、常见问题 FAQ

### Q1: 训练时报错 "STMN.__init__() got unexpected keyword argument 'gate_init'"
✗ 这是Phase 2的旧代码残留。Phase 3的STMN.__init__只有dim和mlp两个参数。
✓ 确认用的是Phase 3版本的代码包。

### Q2: 训练时报错 "AdaMBA n_branches mismatch"
✗ Phase 3所有有AdaMBA的模式都用n_branches=2。
✓ 确认config.py中ada_mba.n_branches=2。

### Q3: gate值一直是0.27没有变化
✓ 这是正常现象！sigmoid(-1)=0.27是初始值。
如果50个epoch后gate仍然≈0.27，说明STMN没有学到有用的东西，模型选择了安全兜底（等同no_stmn）。
这不崩溃，只是STMN贡献为0。

### Q4: (venv)(base) 双环境问题
```bash
deactivate  # 先退出venv
# 然后直接用base环境的python
python code/train.py --mode full
```

### Q5: 系统盘快满了怎么办
```bash
df -h /  # 查看系统盘使用率
# 缓存已重定向到数据盘，一般不会爆
# 如果还是满了：
rm -rf /root/.cache  # 清理系统盘缓存
```

### Q6: 训练卡住不动
✓ 检查GPU是否在用：`nvidia-smi`
✓ 检查进程：`ps aux | grep python`
✓ 如果GPU利用率=0%，可能是数据加载问题

### Q7: 需要重新预处理数据吗？
✓ 只有当 `/root/autodl-tmp/so2sat_exp/data/*.pt` 不存在时才需要
✓ 如果之前的数据还在，**不需要重新预处理**

---

## ★★★ 完成后记得关机停计费！ ★★★

```bash
# 训练完成后，下载结果，然后关机
# 在AutoDL控制台点击"关机"
# ⚠️ 关机后数据盘保留，释放实例后数据丢失
```

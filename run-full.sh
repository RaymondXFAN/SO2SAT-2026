#!/bin/bash
# ★ 论文版（路线A）：完整数据集实验脚本
# ★ 提案模型 = no_stmn (2-分支Ada-MBA + LayerNorm)
# ★ 需要3次不同seed验证no_stmn，1次验证STMN有害
# ★ 使用完整数据集(h5直接读取)，不创建巨大.pt文件
# ★ 预计总耗时约8-10小时
# 用法: cd /root/autodl-tmp/so2sat_exp && nohup bash run-full.sh > paper_log.txt 2>&1 &
# 监控: tail -f /root/autodl-tmp/so2sat_exp/paper_log.txt

set -e
cd /root/autodl-tmp/so2sat_exp

echo "============================================================"
echo "★ ★ 论文版实验 — 完整数据集 (路线A)"
echo "★ 提案模型: 2-分支Ada-MBA + LayerNorm (no_stmn)"
echo "★ 预计总耗时: 8-10小时"
echo "★ 开始时间: $(date)"
echo "============================================================"

# ── Step 1: 确认stats.json存在（完整数据集预处理） ──
if [ ! -f "data/stats.json" ]; then
    echo "✗ stats.json 不存在！先运行预处理..."
    python -u code/prepare_so2sat.py
fi

# ── Step 2: 核心实验 ──

# ★★★ 提案模型: no_stmn × 3 seeds（论文需要均值±std）
echo ""
echo "── ★ 核心实验1: no_stmn × 3 seeds ──"
for SEED in 42 123 456; do
    echo "── 开始: no_stmn seed=$SEED | $(date) ──"
    python -u code/train.py --mode no_stmn --seed $SEED
    echo "── 完成: no_stmn seed=$SEED | $(date) ──"
done

# ★★★ STMN有害性验证: full × 1 seed
echo ""
echo "── ★ 核心实验2: full × 1 seed ──"
python -u code/train.py --mode full --seed 42

# ★★★ Ada-MBA有效性验证
echo ""
echo "── ★ 核心实验3: concat + no_ada ──"
python -u code/train.py --mode concat --seed 42
python -u code/train.py --mode no_ada --seed 42

# ── Step 3: 补充实验 ──
echo ""
echo "── ★ 补充实验: s2_only + s1_only + no_gate ──"
python -u code/train.py --mode s2_only --seed 42
python -u code/train.py --mode s1_only --seed 42
python -u code/train.py --mode no_gate --seed 42

# ── Step 4: 结果汇总 ──
echo ""
echo "============================================================"
echo "★ ★ 论文版实验完成！"
echo "★ 结束时间: $(date)"
echo "============================================================"
echo ""
echo "── ★ no_stmn 多seed结果（提案模型） ──"
for SEED in 42 123 456; do
    DIR="out/no_stmn_seed${SEED}"
    if [ -d "$DIR" ]; then
        oa=$(grep "^oa" "$DIR/table2_main.csv" | cut -d',' -f2)
        echo "  seed=$SEED: OA=$oa%"
    else
        echo "  seed=$SEED: ❌ 未完成"
    fi
done

echo ""
echo "── ★ 其他模式结果 ──"
for MODE in full concat no_ada s2_only s1_only no_gate; do
    DIR="out/$MODE"
    if [ -d "$DIR" ]; then
        oa=$(grep "^oa" "$DIR/table2_main.csv" | cut -d',' -f2)
        echo "  $MODE: OA=$oa%"
    else
        echo "  $MODE: ❌ 未完成"
    fi
done

echo ""
echo "★ 结果文件目录: /root/autodl-tmp/so2sat_exp/out/"
echo "⚠️ 下载结果后请关机停计费！"

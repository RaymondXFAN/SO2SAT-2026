#!/bin/bash
# ★ 论文审稿修复补实验脚本（v3.1）
# ★ 3个Phase按优先级排列，可以随时Ctrl+C停止
# ★ 预计总耗时: Phase1=3小时, Phase2=1.5小时, Phase3=需代码改动

set -e
cd /root/autodl-tmp/so2sat_exp

echo "============================================================"
echo "★ ★ 审稿修复补实验 — 开始时间: $(date)"
echo "★ ★ 当前数据盘状态:"
df -h /root/autodl-tmp | tail -1
echo "============================================================"

# ============================================================
# Phase 1: 320K多seed补完（🔥最关键）
# 目标: 消除"单种子vs多种子"不一致问题
# 已有: full/concat/no_gate/no_ada 各1个seed=42, no_stmn 3 seed
# 需补: full/concat/no_gate/no_ada 各2个seed (123, 456)
# 预计耗时: ~2.5小时
# ============================================================

echo ""
echo "── Phase 1: 320K多seed补完 (最关键) ──"
echo "── 已有seed=42，需要补seed=123和seed=456 ──"

for MODE in full concat no_gate no_ada; do
    for SEED in 123 456; do
        DIR="out/${MODE}_seed${SEED}"
        if [ -f "$DIR/table2_main.csv" ]; then
            echo "✓ $MODE seed=$SEED 已存在，跳过"
            continue
        fi
        echo ""
        echo "── 开始: $MODE seed=$SEED | $(date) ──"
        python -u code/train.py --mode $MODE --seed $SEED
        echo "── 完成: $MODE seed=$SEED | $(date) ──"
    done
done

echo ""
echo "============================================================"
echo "★ Phase 1 完成！"
echo "============================================================"


# ============================================================
# Phase 2: 5K多seed验证
# 目标: 让5K的所有结果都有3 seed均值±std
# 已有: 5K全部7个变体×1 seed
# 需补: 7个变体×2 seeds (123, 456) = 14次跑
# 注意: 需要临时切换到 FULL_DATA=False
# 预计耗时: ~50分钟
# ============================================================

echo ""
echo "── Phase 2: 5K多seed验证 ──"
echo "── 需要临时切换 FULL_DATA=False，跑完恢复 ──"

# 临时切换到5K模式
echo "⚙️ 临时切换 FULL_DATA=True → False"
sed -i.bak 's/^FULL_DATA = True.*/FULL_DATA = False  # TEMP: 5K mode/' code/config.py

# 备份现有5K结果
mkdir -p out/backup_5k_seed42
cp -r out/full out/s2_only out/s1_only out/concat out/no_stmn out/no_ada out/no_gate out/backup_5k_seed42/ 2>/dev/null || echo "⚠️ 部分5K结果未找到"

for MODE in full s2_only s1_only concat no_stmn no_ada no_gate; do
    for SEED in 123 456; do
        # 5K输出目录带 _seed{N} 后缀
        # 由于train.py只在seed≠42时加seed后缀，需要确保123和456走对路径
        DIR="out/${MODE}_seed${SEED}"
        if [ -f "$DIR/table2_main.csv" ]; then
            echo "✓ $MODE 5K seed=$SEED 已存在，跳过"
            continue
        fi
        echo ""
        echo "── 开始: $MODE 5K seed=$SEED | $(date) ──"
        python -u code/train.py --mode $MODE --seed $SEED
        echo "── 完成: $MODE 5K seed=$SEED | $(date) ──"
    done
done

# 恢复 FULL_DATA=True
echo ""
echo "⚙️ 恢复 FULL_DATA=False → True"
mv code/config.py.bak code/config.py

echo ""
echo "============================================================"
echo "★ Phase 2 完成！"
echo "============================================================"


# ============================================================
# Phase 3 (可选): 汇总统计 + 生成 Appendix C 数据
# ============================================================

echo ""
echo "── Phase 3: 汇总所有实验结果 ──"

echo ""
echo "── ★ 5K多seed结果 ──"
for MODE in full s2_only s1_only concat no_stmn no_ada no_gate; do
    echo "  $MODE:"
    for SEED in 42 123 456; do
        if [ "$SEED" = "42" ]; then
            DIR="out/$MODE"
        else
            DIR="out/${MODE}_seed${SEED}"
        fi
        if [ -f "$DIR/table2_main.csv" ]; then
            oa=$(grep "^oa" "$DIR/table2_main.csv" | cut -d',' -f2)
            echo "    seed=$SEED: OA=$oa%"
        fi
    done
done

echo ""
echo "── ★ 320K多seed结果 ──"
for MODE in full concat no_gate no_ada no_stmn; do
    echo "  $MODE:"
    for SEED in 42 123 456; do
        if [ "$SEED" = "42" ]; then
            DIR="out/$MODE"
        else
            DIR="out/${MODE}_seed${SEED}"
        fi
        if [ -f "$DIR/table2_main.csv" ]; then
            oa=$(grep "^oa" "$DIR/table2_main.csv" | cut -d',' -f2)
            echo "    seed=$SEED: OA=$oa%"
        fi
    done
done

echo ""
echo "── ★ Ada-MBA Gate统计 (Appendix C) ──"
echo "5K (full模式):"
if [ -f "out/full/gate_value.txt" ]; then
    cat "out/full/gate_value.txt"
fi
echo ""
echo "320K (full模式):"
if [ -f "out/full/gate_value.txt" ]; then
    cat "out/full/gate_value.txt"
fi

echo ""
echo "============================================================"
echo "★ ★ 全部补实验完成！结束时间: $(date)"
echo "★ 提示: 可以下载结果+tar打包回家了"
echo "============================================================"

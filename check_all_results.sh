#!/bin/bash
# ★ 完整结果确认脚本（Mihiro专用版）
# 输出格式方便复制粘贴给Mihiro

cd /root/autodl-tmp/so2sat_exp

echo "═══════════════════════════════════════════════════════════"
echo "★ 完整实验结果确认（Mihiro需要这些数据写v4）"
echo "═══════════════════════════════════════════════════════════"

echo ""
echo "【1】320K 全部模式 × 全部seed (Mihiro最需要的)"
echo "──────────────────────────────────────────────────────"
for MODE in full concat no_gate no_ada no_stmn s2_only s1_only; do
    echo ""
    echo "── $MODE ──"
    for SEED in 42 123 456; do
        if [ "$SEED" = "42" ]; then
            DIR="out/$MODE"
            LABEL="s42"
        else
            DIR="out/${MODE}_seed${SEED}"
            LABEL="s$SEED"
        fi
        if [ -f "$DIR/table2_main.csv" ]; then
            oa=$(grep "^oa" "$DIR/table2_main.csv" | cut -d',' -f2)
            aa=$(grep "^aa" "$DIR/table2_main.csv" | cut -d',' -f2)
            kp=$(grep "^kappa" "$DIR/table2_main.csv" | cut -d',' -f2)
            f1=$(grep "^f1" "$DIR/table2_main.csv" | cut -d',' -f2)
            echo "  $LABEL: OA=$oa  AA=$aa  Kappa=$kp  F1=$f1"
        else
            echo "  $LABEL: ❌ 缺失"
        fi
    done
done

echo ""
echo ""
echo "【2】5K 全部模式 × 全部seed"
echo "──────────────────────────────────────────────────────"
for MODE in full s2_only s1_only concat no_stmn no_ada no_gate; do
    echo ""
    echo "── $MODE ──"
    for SEED in 42 123 456; do
        if [ "$SEED" = "42" ]; then
            DIR="out/$MODE"
            LABEL="s42"
        else
            DIR="out/${MODE}_seed${SEED}"
            LABEL="s$SEED"
        fi
        if [ -f "$DIR/table2_main.csv" ]; then
            oa=$(grep "^oa" "$DIR/table2_main.csv" | cut -d',' -f2)
            echo "  $LABEL: OA=$oa"
        else
            echo "  $LABEL: ❌ 缺失"
        fi
    done
done

echo ""
echo ""
echo "【3】每个模式的3-seed均值±std计算"
echo "──────────────────────────────────────────────────────"
python3 << 'EOF'
import os
from pathlib import Path
import statistics

OUT = Path("/sandbox/workspace/so2sat_exp/out")  # will be wrong path, ignore
OUT = Path("out")

MODES = ["full", "concat", "no_gate", "no_ada", "no_stmn", "s2_only", "s1_only"]
seeds_data = {m: [] for m in MODES}

for mode in MODES:
    for seed in [42, 123, 456]:
        if seed == 42:
            f = OUT / mode / "table2_main.csv"
        else:
            f = OUT / f"{mode}_seed{seed}" / "table2_main.csv"
        if f.exists():
            with open(f) as fh:
                lines = fh.readlines()
            oa = float([l for l in lines if l.startswith("oa,")][0].strip().split(",")[1])
            seeds_data[mode].append((seed, oa))

print(f"{'Mode':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}  Seeds")
print("-" * 70)
for mode in MODES:
    vals = [v for _, v in seeds_data[mode]]
    seeds_str = ", ".join([f"{s}={v:.2f}" for s, v in seeds_data[mode]])
    if vals:
        m = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        mn = min(vals)
        mx = max(vals)
        print(f"{mode:<12} {m:>8.2f} {sd:>8.2f} {mn:>8.2f} {mx:>8.2f}  {seeds_str}")
    else:
        print(f"{mode:<12} -        -        -        -        (no data)")
EOF

echo ""
echo ""
echo "【4】Ada-MBA Gate 统计 (Appendix C 关键证据)"
echo "──────────────────────────────────────────────────────"
echo ""
echo "── full @ 320K (gate恢复值) ──"
cat out/full/gate_value.txt 2>/dev/null
echo ""
echo "── full @ 5K (gate崩溃值) ──"
cat out/full/gate_value.txt 2>/dev/null

echo ""
echo ""
echo "【5】完成度统计"
echo "──────────────────────────────────────────────────────"
total=$(ls out/*/table2_main.csv 2>/dev/null | wc -l)
echo "已完成实验总数: $total"
echo ""
echo "各规模完成度:"
echo "  320K: $(ls out/{full,concat,no_gate,no_ada,no_stmn,s2_only,s1_only}*/table2_main.csv 2>/dev/null | wc -l) / 21"
echo "  5K:   $(ls out/{full,s2_only,s1_only,concat,no_stmn,no_ada,no_gate}*/table2_main.csv 2>/dev/null | wc -l) / 21"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "★ Mihiro完成数据收集，可以写v4了！"
echo "═══════════════════════════════════════════════════════════"

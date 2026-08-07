#!/bin/bash
# ★ Phase 6: STMN回归独立第3纹理分支 + 3-分支Ada-MBA + LayerNorm
# 一键跑7种模式：full, s2_only, s1_only, concat, no_stmn, no_ada, no_gate
# ★ 使用nohup后台运行，日志输出到phase6_log.txt
# ★ 预计总耗时约3-4小时（7×30-40分钟）
# 用法: cd /root/autodl-tmp/so2sat_exp && nohup bash run-all.sh > phase6_log.txt 2>&1 &
# 监控: tail -f /root/autodl-tmp/so2sat_exp/phase6_log.txt

set -e  # 遇到错误停止
cd /root/autodl-tmp/so2sat_exp

echo "============================================================"
echo "★ Phase 6 实验开始 — STMN独立第3纹理分支 + 3-分支Ada-MBA"
echo "★ 预计总耗时: 3-4小时"
echo "★ 开始时间: $(date)"
echo "============================================================"

# ── 核心模式（按重要性排列） ──
MODES=("full" "no_stmn" "no_gate" "no_ada" "s2_only" "s1_only" "concat")

for mode in "${MODES[@]}"; do
    echo ""
    echo "── ★ 开始训练: $mode ──"
    echo "── 开始时间: $(date) ──"
    python -u code/train.py --mode $mode
    echo "── ★ 完成: $mode ──"
    echo "── 完成时间: $(date) ──"
done

echo ""
echo "============================================================"
echo "★ Phase 6 全部7种模式实验完成！"
echo "★ 结束时间: $(date)"
echo "============================================================"
echo ""
echo "── ★ 结果汇总 ──"
for mode in "${MODES[@]}"; do
    if [ -f "out/$mode/table2_main.csv" ]; then
        oa=$(grep "^oa" "out/$mode/table2_main.csv" | cut -d',' -f2)
        echo "  $mode: OA=$oa%"
    else
        echo "  $mode: 结果文件不存在！"
    fi
done

echo ""
echo "── ★ Gate值（full模式） ──"
if [ -f "out/full/gate_value.txt" ]; then
    cat out/full/gate_value.txt
else
    echo "  gate_value.txt 不存在！"
fi

echo ""
echo "★ 结果文件目录: /root/autodl-tmp/so2sat_exp/out/"
echo "⚠️ 下载结果后请关机停计费！"

#!/bin/bash
# 精简打包脚本 - 只含代码和核心结果，不含大数据
# 用法：在云端 AutoDL 执行 bash package_lite.sh

PROJECT_DIR="/root/autodl-tmp/HierFed-Matter-NSAC-DPBA"
cd "$PROJECT_DIR" || { echo "❌ 项目目录不存在: $PROJECT_DIR"; exit 1; }

OUT_NAME="HierFed-NSAC-DPBA-code-results-$(date +%Y%m%d)"
OUT_FILE="/root/autodl-tmp/${OUT_NAME}.tar.gz"

# ① 先看每个目录多大
echo "📊 各目录大小（前15）："
du -sh */ 2>/dev/null | sort -hr | head -15

echo ""
echo "🗜️  开始打包（排除数据/缓存/日志）..."

# ② 精简打包：只含代码+核心结果
tar --exclude="data/raw/extracted" \
    --exclude="data/processed/*.npz" \
    --exclude="data/processed/*partitions*.json" \
    --exclude="__pycache__" \
    --exclude="**/__pycache__" \
    --exclude=".ipynb_checkpoints" \
    --exclude="*.pyc" \
    --exclude="*.log" \
    --exclude="recover_log.txt" \
    --exclude="epsweep_log.txt" \
    --exclude="baseline_log.txt" \
    --exclude="results/baseline_*" \
    --exclude="results/epsweep_recovery" \
    --exclude="results/summary/epsweep_*recovery*" \
    -czf "$OUT_FILE" \
    core/ models/ configs/ baselines/ \
    run_*.py plot_*.py setup_*.sh run_all_*.sh \
    requirements.txt README.md \
    results/epsweep_eps*/ \
    results/summary/ \
    data/ 2>/dev/null

echo ""
echo "✅ 打包完成！"
ls -lh "$OUT_FILE"
echo ""

# ③ 看包内文件清单
echo "📦 包内文件（前30个）："
tar -tzf "$OUT_FILE" | head -30
echo ""
echo "📊 包内文件总数: $(tar -tzf "$OUT_FILE" | wc -l)"

echo ""
echo "💡 下一步："
echo "  1. 在 AutoDL 控制台 → 文件管理 → 下载 ${OUT_NAME}.tar.gz"
echo "  2. 或用 scp/rsync 下载"
echo "  3. 解压后跑: python3 plot_privacy_utility.py 看图"
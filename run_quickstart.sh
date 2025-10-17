#!/bin/bash

# LLM微调快速体验脚本
# 一键运行完整流程

set -e  # 遇到错误就退出

echo "🚀 LLM微调全流程快速体验"
echo "================================"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ 未找到Python，请先安装Python"
    exit 1
fi

# 安装依赖（如果需要）
if [ ! -f "venv/bin/activate" ]; then
    echo "📦 创建虚拟环境..."
    python -m venv venv
fi

echo "🔧 激活虚拟环境..."
source venv/bin/activate

echo "📦 安装依赖包..."
pip install -r requirements.txt --quiet

echo "🎯 开始全流程体验！"
echo ""

# Step 1: 数据准备
echo "📊 Step 1: 准备训练数据..."
python scripts/prepare_data.py
echo ""

# Step 2: 模型训练
echo "🏋️ Step 2: 开始微调训练..."
echo "⏰ 预计用时: 10-15分钟（取决于硬件配置）"
python scripts/train.py
echo ""

# Step 3: 模型评估
echo "📈 Step 3: 评估模型性能..."
python scripts/evaluate.py
echo ""

# Step 4: 启动推理服务
echo "🌐 Step 4: 启动推理服务..."
echo "🎉 训练完成！即将启动Web服务..."
echo "📱 访问地址: http://localhost:7860"
echo "💡 按 Ctrl+C 停止服务"
echo ""

python scripts/inference.py --chat --share

echo "✅ 全流程体验完成！"
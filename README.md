# 🚀 LLM微调全流程快速体验（2-4小时）

一个完整的大语言模型微调项目，包含数据准备、模型训练、评估和部署的全流程实践。

## 📋 项目概览

本项目提供了一个快速体验LLM微调的完整流程，适合初学者在2-4小时内完成：

- **数据准备**：自动生成训练数据集
- **微调训练**：使用LoRA高效微调
- **模型评估**：多维度评估模型性能
- **部署推理**：Web界面交互式体验

## 🛠️ 环境准备

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖包
pip install -r requirements.txt
```

### 2. 硬件要求

- **最低配置**：8GB 显存（使用4bit量化）
- **推荐配置**：16GB+ 显存
- **CPU运行**：支持但速度较慢

## 🎯 快速开始（15分钟）

### Step 1: 准备数据（2分钟）

```bash
cd llm-finetune-quickstart
python scripts/prepare_data.py
```

这将生成：
- `data/train.jsonl` - 训练数据集
- `data/eval.jsonl` - 验证数据集
- 包含问答、翻译、代码生成等多种任务类型

### Step 2: 开始训练（10分钟）

```bash
python scripts/train.py
```

**训练配置说明**：
- 使用 Qwen2-0.5B 小模型（快速体验）
- LoRA微调（参数效率高）
- 3个epoch（约10分钟完成）
- 自动保存检查点

### Step 3: 评估模型（2分钟）

```bash
python scripts/evaluate.py
```

评估指标包括：
- 困惑度（Perplexity）
- 生成质量
- 响应完整性
- 词汇多样性

### Step 4: 部署推理（1分钟）

```bash
# 启动Web服务
python scripts/inference.py

# 或使用聊天界面
python scripts/inference.py --chat

# 生成公共链接（可选）
python scripts/inference.py --share
```

访问 http://localhost:7860 即可体验！

## 📁 项目结构

```
llm-finetune-quickstart/
├── configs/
│   └── training_config.yaml    # 训练配置文件
├── data/
│   ├── train.jsonl             # 训练数据
│   └── eval.jsonl              # 验证数据
├── scripts/
│   ├── prepare_data.py         # 数据准备
│   ├── train.py                # 训练脚本
│   ├── evaluate.py             # 评估脚本
│   └── inference.py            # 推理部署
├── outputs/
│   ├── qwen-lora/              # 模型输出
│   └── evaluation/             # 评估结果
└── requirements.txt            # 依赖列表
```

## 🔧 进阶配置

### 自定义训练参数

编辑 `configs/training_config.yaml`：

```yaml
# 更换基础模型
model:
  name: "meta-llama/Llama-2-7b-hf"  # 使用更大的模型

# 调整训练参数
training:
  num_train_epochs: 5              # 增加训练轮数
  learning_rate: 1e-4              # 调整学习率
  per_device_train_batch_size: 8   # 增大批次（需要更多显存）
```

### 使用自己的数据

修改 `scripts/prepare_data.py` 或直接准备 JSONL 格式数据：

```json
{"instruction": "你的指令", "input": "可选输入", "output": "期望输出"}
```

### 高级推理参数

```bash
python scripts/inference.py \
  --model-path ./outputs/qwen-lora \
  --base-model Qwen/Qwen2-0.5B \
  --port 8080 \
  --chat
```

## 💡 常见问题

### 1. 显存不足

**解决方案**：
- 启用4bit量化（默认已开启）
- 减小批次大小
- 使用梯度累积
- 选择更小的模型

### 2. 训练速度慢

**优化建议**：
- 使用GPU训练
- 开启fp16混合精度
- 减少max_length
- 使用更小的模型

### 3. 生成质量差

**改进方法**：
- 增加训练数据
- 调整学习率
- 增加训练轮数
- 优化数据质量

## 🚀 下一步

完成快速体验后，可以尝试：

1. **扩展数据集**：使用更多领域的数据
2. **尝试其他模型**：ChatGLM、Baichuan、Llama等
3. **优化训练策略**：调整超参数、使用不同的优化器
4. **部署优化**：量化、剪枝、模型压缩
5. **构建应用**：集成到实际业务场景

## 📚 学习资源

- [Hugging Face Transformers文档](https://huggingface.co/docs/transformers)
- [PEFT库文档](https://github.com/huggingface/peft)
- [LLM微调最佳实践](https://github.com/huggingface/alignment-handbook)

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

---

**提示**：这是一个教学项目，生产环境部署需要更多优化和安全考虑。
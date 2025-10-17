#!/usr/bin/env python3
"""
数据准备脚本 - 生成指令微调数据集
"""

import json
import random
from pathlib import Path

def create_sample_dataset():
    """创建示例指令数据集"""
    
    # 示例训练数据 - 包含各种任务类型
    train_samples = [
        # 问答类
        {"instruction": "什么是机器学习？", "output": "机器学习是人工智能的一个分支，它使计算机能够通过数据学习模式和规律，而无需明确编程。"},
        {"instruction": "Python和Java的主要区别是什么？", "output": "Python是解释型语言，语法简洁，适合快速开发；Java是编译型语言，性能更好，适合大型企业应用。"},
        {"instruction": "解释一下深度学习的概念", "output": "深度学习是机器学习的子领域，使用多层神经网络来学习数据的复杂模式和表示。"},
        
        # 翻译类
        {"instruction": "将下面的句子翻译成英文：今天天气很好", "output": "The weather is very nice today."},
        {"instruction": "Translate to Chinese: Artificial Intelligence is changing the world", "output": "人工智能正在改变世界。"},
        
        # 摘要类
        {"instruction": "总结以下内容：云计算是一种通过互联网提供计算资源的模式，包括服务器、存储、数据库等。它具有按需使用、弹性扩展、成本效益高等特点。", 
         "output": "云计算是通过互联网提供可按需使用、弹性扩展的计算资源服务模式。"},
        
        # 代码生成
        {"instruction": "写一个Python函数计算列表的平均值", "output": "def calculate_average(numbers):\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)"},
        {"instruction": "用Python实现冒泡排序", "output": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr"},
        
        # 分类任务
        {"instruction": "判断以下评论的情感倾向：这个产品质量很好，我很满意", "output": "正面"},
        {"instruction": "分类以下文本的主题：神经网络是由大量神经元相互连接形成的网络结构", "output": "技术/人工智能"},
        
        # 创意写作
        {"instruction": "续写故事：在一个遥远的星球上，", "output": "在一个遥远的星球上，生活着一群拥有独特能力的生物。他们能够通过思维直接交流，创造出美妙的光影艺术。"},
        {"instruction": "为一家咖啡店写一句广告语", "output": "醇香每一刻，温暖每颗心 - 您的专属咖啡时光"},
        
        # 数学推理
        {"instruction": "解决问题：如果一个班级有30名学生，其中60%是女生，那么男生有多少人？", "output": "男生人数 = 30 × (1 - 60%) = 30 × 0.4 = 12人"},
        {"instruction": "计算：(15 + 25) × 3 - 20", "output": "(15 + 25) × 3 - 20 = 40 × 3 - 20 = 120 - 20 = 100"},
        
        # 信息提取
        {"instruction": "从以下文本中提取人名：张三是一位优秀的工程师，他和李四一起完成了这个项目", "output": "张三、李四"},
        {"instruction": "提取关键词：机器学习模型需要大量数据进行训练才能获得良好的性能", "output": "机器学习、模型、数据、训练、性能"},
    ]
    
    # 扩展数据集 - 通过变体生成更多样本
    extended_samples = []
    for sample in train_samples:
        extended_samples.append(sample)
        # 添加带输入的版本
        if random.random() > 0.5:
            extended_samples.append({
                "instruction": sample["instruction"],
                "input": f"请回答：{sample['instruction']}",
                "output": sample["output"]
            })
    
    # 打乱顺序
    random.shuffle(extended_samples)
    
    # 分割训练集和验证集 (80/20)
    split_idx = int(len(extended_samples) * 0.8)
    train_data = extended_samples[:split_idx]
    eval_data = extended_samples[split_idx:]
    
    # 保存数据
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    with open(data_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(data_dir / "eval.jsonl", "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ 数据准备完成！")
    print(f"   训练集: {len(train_data)} 条")
    print(f"   验证集: {len(eval_data)} 条")
    print(f"   保存位置: ./data/")
    
    return len(train_data), len(eval_data)

def format_prompts(data_file, output_file):
    """将数据格式化为对话格式"""
    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            
            # 构建对话格式
            if "input" in item and item["input"]:
                prompt = f"### 指令：\n{item['instruction']}\n\n### 输入：\n{item['input']}\n\n### 回答：\n{item['output']}"
            else:
                prompt = f"### 指令：\n{item['instruction']}\n\n### 回答：\n{item['output']}"
            
            samples.append({"text": prompt})
    
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"✅ 格式化完成: {output_file}")

if __name__ == "__main__":
    print("🚀 开始准备数据...")
    
    # 创建数据集
    train_size, eval_size = create_sample_dataset()
    
    # 格式化数据
    format_prompts("./data/train.jsonl", "./data/train_formatted.jsonl")
    format_prompts("./data/eval.jsonl", "./data/eval_formatted.jsonl")
    
    print("\n📊 数据统计：")
    print(f"   总样本数: {train_size + eval_size}")
    print(f"   数据格式: JSONL (指令-输出对)")
    print("\n💡 提示：可以根据需要修改此脚本来加载自己的数据集")
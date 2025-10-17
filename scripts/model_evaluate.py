#!/usr/bin/env python3
"""
模型评估脚本 - 评估微调后的模型性能
"""

import json
import torch
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
# import evaluate  # 暂时注释掉，避免依赖问题
from tqdm import tqdm
import numpy as np
from collections import Counter

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_and_tokenizer(model_path, base_model_name=None):
    """加载微调后的模型"""
    print("📥 加载模型...")
    
    if base_model_name:
        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        # 加载LoRA权重
        model = PeftModel.from_pretrained(model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    else:
        # 直接加载合并后的模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512):
    """生成模型响应"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取生成的部分
    response = response[len(prompt):].strip()
    return response

def evaluate_perplexity(model, tokenizer, dataset, num_samples=100):
    """计算困惑度"""
    print("\n📊 计算困惑度...")
    
    total_loss = 0
    total_tokens = 0
    
    for i in tqdm(range(min(num_samples, len(dataset)))):
        text = dataset[i]['text']
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()

def evaluate_generation_quality(model, tokenizer, test_prompts):
    """评估生成质量"""
    print("\n🎯 评估生成质量...")
    results = []
    
    for prompt in tqdm(test_prompts):
        response = generate_response(model, tokenizer, prompt['instruction'])
        results.append({
            'instruction': prompt['instruction'],
            'generated': response,
            'expected': prompt.get('output', '')
        })
    
    return results

def calculate_metrics(results):
    """计算评估指标"""
    print("\n📈 计算评估指标...")
    
    # 基础统计
    total = len(results)
    avg_length = np.mean([len(r['generated'].split()) for r in results])
    
    # 响应完整性（是否生成了内容）
    non_empty = sum(1 for r in results if len(r['generated'].strip()) > 0)
    completeness_rate = non_empty / total if total > 0 else 0
    
    # 多样性评估（unique n-grams）
    all_tokens = []
    for r in results:
        all_tokens.extend(r['generated'].split())
    
    unique_unigrams = len(set(all_tokens))
    total_unigrams = len(all_tokens)
    unigram_diversity = unique_unigrams / total_unigrams if total_unigrams > 0 else 0
    
    # 简单的文本相似度评估（如果有参考答案）
    similarity_scores = []
    
    for r in results:
        if r['expected']:
            # 简单的词汇重叠度计算
            gen_words = set(r['generated'].lower().split())
            exp_words = set(r['expected'].lower().split())
            
            if len(exp_words) > 0:
                overlap = len(gen_words.intersection(exp_words))
                similarity = overlap / len(exp_words)
                similarity_scores.append(similarity)
    
    avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
    
    metrics = {
        'total_samples': total,
        'completeness_rate': completeness_rate,
        'avg_response_length': avg_length,
        'unigram_diversity': unigram_diversity,
        'avg_similarity': avg_similarity
    }
    
    return metrics

def main():
    # 加载配置
    config = load_config('./configs/training_config.yaml')
    
    # 模型路径
    model_path = config['training']['output_dir']
    base_model = config['model']['name']
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(model_path, base_model)
    
    # 加载测试数据
    print("\n📂 加载测试数据...")
    eval_dataset = load_dataset(
        'json',
        data_files={'eval': './data/eval_formatted.jsonl'}
    )['eval']
    
    # 测试提示
    test_prompts = [
        {"instruction": "什么是深度学习？", "output": "深度学习是机器学习的子领域"},
        {"instruction": "写一个Python函数计算斐波那契数列", "output": ""},
        {"instruction": "将下面的句子翻译成英文：人工智能正在改变世界", "output": ""},
        {"instruction": "总结以下内容：大语言模型是基于Transformer架构的模型", "output": ""},
        {"instruction": "判断情感倾向：这个产品真的很棒，我很喜欢", "output": "正面"},
    ]
    
    # 评估困惑度
    perplexity = evaluate_perplexity(model, tokenizer, eval_dataset, num_samples=50)
    
    # 评估生成质量
    generation_results = evaluate_generation_quality(model, tokenizer, test_prompts)
    
    # 计算指标
    metrics = calculate_metrics(generation_results)
    
    # 打印结果
    print("\n" + "="*50)
    print("📊 评估结果汇总")
    print("="*50)
    print(f"困惑度 (Perplexity): {perplexity:.2f}")
    print(f"完成率: {metrics['completeness_rate']*100:.1f}%")
    print(f"平均响应长度: {metrics['avg_response_length']:.1f} 词")
    print(f"词汇多样性: {metrics['unigram_diversity']:.3f}")
    if metrics['avg_similarity'] > 0:
        print(f"文本相似度: {metrics['avg_similarity']:.3f}")
    
    print("\n📝 生成示例:")
    print("-"*50)
    for i, result in enumerate(generation_results[:3], 1):
        print(f"\n示例 {i}:")
        print(f"指令: {result['instruction']}")
        print(f"生成: {result['generated'][:200]}...")
        if result['expected']:
            print(f"参考: {result['expected'][:200]}...")
    
    # 保存评估结果
    output_dir = Path('./outputs/evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'perplexity': perplexity,
            'generation_samples': generation_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 评估完成！结果已保存到: {output_dir / 'evaluation_results.json'}")

if __name__ == "__main__":
    main()
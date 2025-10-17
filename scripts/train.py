#!/usr/bin/env python3
"""
微调训练脚本 - 使用LoRA进行高效微调
"""

import os
import sys
import yaml
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_model_and_tokenizer(config):
    """准备模型和分词器"""
    print("📥 加载模型和分词器...")
    
    # 检测运行环境
    use_cuda = torch.cuda.is_available()
    use_quantization = use_cuda and config['model']['load_in_4bit']
    
    print(f"   CUDA可用: {use_cuda}")
    print(f"   使用量化: {use_quantization}")
    
    # 量化配置（仅在CUDA可用时）
    bnb_config = None
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        quantization_config=bnb_config,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
        torch_dtype=torch.float16 if use_cuda else torch.float32
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        trust_remote_code=True
    )
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 准备模型进行训练
    if use_quantization:
        model = prepare_model_for_kbit_training(model)
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        task_type=TaskType[config['lora']['task_type']],
        bias="none"
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def preprocess_function(examples, tokenizer, max_length):
    """预处理数据"""
    # 对文本进行分词
    result = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    result["labels"] = result["input_ids"].copy()
    return result

def load_and_prepare_dataset(config, tokenizer):
    """加载和准备数据集"""
    print("📊 加载数据集...")
    
    # 加载数据
    dataset = load_dataset(
        'json',
        data_files={
            'train': config['data']['train_file'].replace('.jsonl', '_formatted.jsonl'),
            'eval': config['data']['eval_file'].replace('.jsonl', '_formatted.jsonl')
        }
    )
    
    # 预处理数据
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, config['data']['max_length']),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_dataset

def main():
    # 加载配置
    config = load_config('./configs/training_config.yaml')
    
    # 准备模型和分词器
    model, tokenizer = prepare_model_and_tokenizer(config)
    
    # 准备数据集
    dataset = load_and_prepare_dataset(config, tokenizer)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=float(config['training']['learning_rate']),
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_steps=config['training']['eval_steps'],
        save_total_limit=config['training']['save_total_limit'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        eval_strategy=config['training']['evaluation_strategy'],
        fp16=config['training']['fp16'] and torch.cuda.is_available(),
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        report_to=config['training']['report_to'],
        logging_dir=f"{config['training']['output_dir']}/logs",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # 数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("\n🚀 开始训练...")
    print(f"   训练样本数: {len(dataset['train'])}")
    print(f"   验证样本数: {len(dataset['eval'])}")
    print(f"   训练轮数: {config['training']['num_train_epochs']}")
    print(f"   批次大小: {config['training']['per_device_train_batch_size']}")
    print(f"   学习率: {config['training']['learning_rate']}")
    print(f"   输出目录: {config['training']['output_dir']}")
    
    # 开始训练
    trainer.train()
    
    # 保存最终模型
    print("\n💾 保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(config['training']['output_dir'])
    
    print(f"\n✅ 训练完成！模型已保存到: {config['training']['output_dir']}")
    
    # 显示训练统计
    if trainer.state.log_history:
        final_stats = trainer.state.log_history[-1]
        print("\n📈 训练统计:")
        if 'loss' in final_stats:
            print(f"   最终训练损失: {final_stats['loss']:.4f}")
        if 'eval_loss' in final_stats:
            print(f"   最终验证损失: {final_stats['eval_loss']:.4f}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
对比原始模型 - 验证微调效果
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_original_model():
    """测试原始Qwen2-0.5B模型"""
    print("📥 加载原始 Qwen2-0.5B 模型...")
    
    # 加载原始模型
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2-0.5B',
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B', trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ 原始模型加载完成")
    
    # 测试问题
    test_prompts = [
        "什么是机器学习？",
        "用Python写一个计算阶乘的函数",
        "将这句话翻译成英文：今天天气很好"
    ]
    
    print("\n" + "="*60)
    print("🧪 原始模型测试")
    print("="*60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 测试 {i}: {prompt}")
        print("-" * 40)
        
        # 格式化提示
        formatted_prompt = f"### 指令：\n{prompt}\n\n### 回答：\n"
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True)
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(formatted_prompt):].strip()
        
        print(f"🤖 原始模型: {response[:200]}...")
    
    print(f"\n💡 对比方法:")
    print(f"   现在运行你的微调模型:")
    print(f"   python scripts/simple_inference.py")
    print(f"   用相同问题测试，对比回答质量！")

if __name__ == "__main__":
    test_original_model()
#!/usr/bin/env python3
"""
简化推理脚本 - 命令行交互，避免Gradio依赖问题
"""

import os
import torch
import yaml
from typing import Optional
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

class SimpleInferenceEngine:
    """简单推理引擎"""
    
    def __init__(self, model_path, base_model_name=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self.load_model(model_path, base_model_name)
        
    def load_model(self, model_path, base_model_name=None):
        """加载模型"""
        print(f"📥 加载模型: {model_path}")
        
        if base_model_name and os.path.exists(model_path):
            # 加载基础模型 + LoRA
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True
            )
            model = PeftModel.from_pretrained(model, model_path)
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            print("✅ 加载微调模型（LoRA）")
        else:
            # 直接加载完整模型或使用基础模型
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map=None,
                    trust_remote_code=True
                )
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                print("✅ 加载完整模型")
            except:
                print("⚠️ 未找到微调模型，使用基础模型")
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float32,
                    device_map=None,
                    trust_remote_code=True
                )
                tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model.eval()
        print(f"✅ 模型加载完成！使用设备: {self.device}")
        return model, tokenizer
    
    def generate(self, 
                prompt: str,
                max_new_tokens: int = 256,
                temperature: float = 0.7,
                top_p: float = 0.9,
                top_k: int = 50):
        """生成响应"""
        
        # 构建输入
        formatted_prompt = f"### 指令：\n{prompt}\n\n### 回答：\n"
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # 生成
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(formatted_prompt):].strip()
        
        generation_time = time.time() - start_time
        tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
        
        return {
            'response': response,
            'generation_time': f"{generation_time:.2f}s",
            'tokens_generated': tokens_generated,
            'tokens_per_second': f"{tokens_generated/generation_time:.1f}"
        }

def interactive_chat(engine):
    """交互式聊天"""
    print("\n" + "="*60)
    print("🤖 LLM微调模型推理演示")
    print("="*60)
    print("💡 输入你的问题，输入 'quit' 或 'exit' 退出")
    print("💡 输入 'clear' 清屏")
    print("💡 输入 'help' 查看帮助")
    print("-"*60)
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n🙋 你: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break
            elif user_input.lower() in ['clear', '清屏']:
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            elif user_input.lower() in ['help', '帮助']:
                print("\n📚 使用说明:")
                print("  - 直接输入问题即可")
                print("  - 'quit/exit' : 退出程序")
                print("  - 'clear' : 清屏")
                print("  - 'help' : 显示帮助")
                continue
            elif not user_input:
                continue
            
            # 生成回答
            print("\n🤖 正在思考...")
            result = engine.generate(user_input)
            
            # 显示结果
            print(f"\n🤖 助手: {result['response']}")
            print(f"\n📊 生成统计: {result['generation_time']} | {result['tokens_generated']} tokens | {result['tokens_per_second']} tokens/s")
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 出错了: {e}")

def batch_test(engine):
    """批量测试"""
    test_cases = [
        "什么是机器学习？",
        "用Python写一个计算阶乘的函数",
        "将这句话翻译成英文：今天天气很好",
        "解释一下什么是深度学习",
        "为什么天空是蓝色的？"
    ]
    
    print("\n" + "="*60)
    print("🧪 批量测试模式")
    print("="*60)
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n📝 测试 {i}/{len(test_cases)}: {prompt}")
        print("-" * 40)
        
        result = engine.generate(prompt, max_new_tokens=150)
        print(f"🤖 回答: {result['response']}")
        print(f"📊 统计: {result['generation_time']} | {result['tokens_per_second']} tokens/s")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="简化LLM推理工具")
    parser.add_argument("--model-path", type=str, default="./outputs/qwen-lora",
                       help="模型路径")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2-0.5B",
                       help="基础模型名称")
    parser.add_argument("--batch", action="store_true",
                       help="批量测试模式")
    
    args = parser.parse_args()
    
    # 初始化推理引擎
    print("🚀 启动推理引擎...")
    engine = SimpleInferenceEngine(args.model_path, args.base_model)
    
    if args.batch:
        batch_test(engine)
    else:
        interactive_chat(engine)

if __name__ == "__main__":
    main()
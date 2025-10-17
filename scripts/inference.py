#!/usr/bin/env python3
"""
推理部署脚本 - 提供Web界面和API服务
"""

import os
import torch
import yaml
import json
import gradio as gr
from typing import Optional
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, model_path, base_model_name=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self.load_model(model_path, base_model_name)
        
    def load_model(self, model_path, base_model_name=None):
        """加载模型"""
        print(f"📥 加载模型: {model_path}")
        
        if base_model_name:
            # 加载基础模型 + LoRA
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            model = PeftModel.from_pretrained(model, model_path)
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        else:
            # 直接加载完整模型
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
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
                top_k: int = 50,
                do_sample: bool = True):
        """生成响应"""
        
        # 构建输入
        formatted_prompt = f"### 指令：\n{prompt}\n\n### 回答：\n"
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 生成
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
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

def create_gradio_interface(engine):
    """创建Gradio Web界面"""
    
    def chat_function(message, history, temperature, max_tokens, top_p):
        """聊天函数"""
        result = engine.generate(
            prompt=message,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        response = result['response']
        stats = f"\n\n---\n⏱️ 生成时间: {result['generation_time']} | 📝 Token数: {result['tokens_generated']} | ⚡ 速度: {result['tokens_per_second']} tokens/s"
        
        return response + stats
    
    # 创建界面
    demo = gr.ChatInterface(
        fn=chat_function,
        title="🤖 LLM微调模型推理演示",
        description="与微调后的模型进行对话。可以调整生成参数来控制输出质量。",
        examples=[
            "什么是机器学习？",
            "用Python写一个快速排序算法",
            "将下面的句子翻译成英文：今天天气很好",
            "帮我写一首关于春天的诗",
            "解释一下什么是区块链技术",
        ],
        additional_inputs=[
            gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature (创造性)"),
            gr.Slider(minimum=50, maximum=500, value=256, step=50, label="Max Tokens (最大长度)"),
            gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.1, label="Top-p (多样性)"),
        ],
        theme=gr.themes.Soft(),
    )
    
    return demo

def create_simple_interface(engine):
    """创建简单界面"""
    
    def inference_function(
        prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=0.9,
        top_k=50
    ):
        """推理函数"""
        if not prompt:
            return "请输入提示词", ""
        
        result = engine.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        
        stats = f"""
### 📊 生成统计
- ⏱️ 生成时间: {result['generation_time']}
- 📝 生成Token数: {result['tokens_generated']}
- ⚡ 生成速度: {result['tokens_per_second']} tokens/s
"""
        
        return result['response'], stats
    
    # 创建界面
    with gr.Blocks(title="LLM推理部署", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 LLM微调模型推理部署")
        gr.Markdown("输入指令，模型将生成相应的回答。")
        
        with gr.Row():
            with gr.Column(scale=6):
                prompt_input = gr.Textbox(
                    label="输入指令",
                    placeholder="例如：什么是深度学习？",
                    lines=3
                )
                
                with gr.Row():
                    with gr.Column():
                        temperature = gr.Slider(
                            minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                            label="Temperature (控制创造性)"
                        )
                        max_tokens = gr.Slider(
                            minimum=50, maximum=500, value=256, step=50,
                            label="Max Tokens (最大生成长度)"
                        )
                    
                    with gr.Column():
                        top_p = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.9, step=0.1,
                            label="Top-p (核采样)"
                        )
                        top_k = gr.Slider(
                            minimum=10, maximum=100, value=50, step=10,
                            label="Top-k (候选词数量)"
                        )
                
                submit_btn = gr.Button("🎯 生成回答", variant="primary")
                
            with gr.Column(scale=6):
                output_text = gr.Textbox(
                    label="模型输出",
                    lines=10
                )
                stats_text = gr.Markdown(label="生成统计")
        
        gr.Examples(
            examples=[
                ["什么是深度学习？"],
                ["用Python实现冒泡排序"],
                ["写一个关于人工智能的短故事"],
                ["解释什么是区块链"],
                ["如何学习编程？"],
            ],
            inputs=prompt_input
        )
        
        submit_btn.click(
            fn=inference_function,
            inputs=[prompt_input, temperature, max_tokens, top_p, top_k],
            outputs=[output_text, stats_text]
        )
    
    return demo

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLM推理部署")
    parser.add_argument("--model-path", type=str, default="./outputs/qwen-lora",
                       help="模型路径")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2-0.5B",
                       help="基础模型名称（如果使用LoRA）")
    parser.add_argument("--port", type=int, default=7860,
                       help="服务端口")
    parser.add_argument("--share", action="store_true",
                       help="生成公共链接")
    parser.add_argument("--chat", action="store_true",
                       help="使用聊天界面")
    
    args = parser.parse_args()
    
    # 初始化推理引擎
    print("🚀 启动推理服务...")
    engine = InferenceEngine(args.model_path, args.base_model)
    
    # 创建界面
    if args.chat:
        demo = create_gradio_interface(engine)
    else:
        demo = create_simple_interface(engine)
    
    # 启动服务
    print(f"\n✅ 服务启动成功！")
    print(f"   本地访问: http://localhost:{args.port}")
    if args.share:
        print(f"   生成公共链接中...")
    
    demo.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0"
    )

if __name__ == "__main__":
    main()
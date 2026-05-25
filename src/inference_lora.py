"""
本地加载微调后的 Qwen2.5-1.5B + LoRA Adapter 跑推理
对比微调前后的回答质量

环境:
- Mac M4 + 16GB 统一内存
- MPS 加速
- 首次运行会从 HuggingFace 下载 base 模型 (~3GB)
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ==================== 配置 ====================
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_DIR = "./models/qwen2.5-1.5b-lora-final"


def get_device():
    """
    Mac M 芯片的 MPS 在 Qwen2.5 GQA 注意力上有兼容性 bug
    (mps_matmul incompatible dimensions for GQA)
    强制使用 CPU - 慢但稳定
    """
    return "cpu"


def load_model(load_lora: bool = True):
    """加载 base model + (可选) LoRA adapter"""
    device = get_device()
    print(f"🖥️  设备: {device}")
    
    print(f"\n📥 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    print(f"📥 加载 base model: {BASE_MODEL}")
    print(f"   首次会下载约 3GB, 后续直接用缓存")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,  # FP16 节省内存
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    if load_lora:
        print(f"🔧 加载 LoRA adapter: {LORA_DIR}")
        model = PeftModel.from_pretrained(model, LORA_DIR)
        print("✅ LoRA adapter 已加载")
    else:
        print("ℹ️  跳过 LoRA, 用纯 base model")
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device


def generate(model, tokenizer, device, question: str, max_new_tokens: int = 300):
    """单次生成"""
    prompt = (
        f"<|im_start|>system\n你是一个 helpful 的中文 AI 助手。<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 抽取 assistant 部分
    answer_start = full_text.find("<|im_start|>assistant\n")
    if answer_start != -1:
        answer = full_text[answer_start + len("<|im_start|>assistant\n"):]
        answer = answer.replace("<|im_end|>", "").strip()
        # 截断到 <|endoftext|> 之类的 token
        for stop_token in ["<|endoftext|>", "<|im_start|>"]:
            idx = answer.find(stop_token)
            if idx != -1:
                answer = answer[:idx].strip()
    else:
        answer = full_text
    
    return answer


# ==================== 测试入口 ====================
if __name__ == "__main__":
    # 测试问题: 跟 Colab 上的一致
    test_questions = [
        "请用 3 句话介绍一下人工智能。",
        "写一首关于秋天的短诗(5 行以内)。",
        "用 Python 写一个判断闰年的函数。",
        "解释一下什么是机器学习,语言要通俗易懂。",
        "用创意的方式描述早晨的阳光。",
    ]
    
    print("=" * 70)
    print("🧪 本地加载微调后模型测试")
    print("=" * 70)
    
    # 加载模型 (含 LoRA)
    model, tokenizer, device = load_model(load_lora=True)
    
    print("\n" + "=" * 70)
    print("📊 开始生成 (每题预计 10-30 秒)")
    print("=" * 70)
    
    import time
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"❓ 问题 {i}: {question}")
        print('='*70)
        
        start = time.time()
        answer = generate(model, tokenizer, device, question)
        elapsed = time.time() - start
        
        print(f"\n💬 回答 ({elapsed:.1f}秒):")
        print(answer)
    
    print(f"\n{'='*70}")
    print("✅ 测试完成!")
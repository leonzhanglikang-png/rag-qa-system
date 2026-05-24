"""
HyDE (Hypothetical Document Embeddings) 查询改写模块

论文: Precise Zero-Shot Dense Retrieval without Relevance Labels
     Luyu Gao et al. (在你的知识库里就有: hyde.pdf)

核心思路:
- 直接用用户查询检索时, 查询是"问题风格", 答案 chunk 是"陈述风格"
- 向量空间里两种风格距离较远 → 召回不准
- HyDE: 先用 LLM 假装回答查询, 用假设答案去检索
        → 假设答案是"陈述风格", 跟真实答案 chunk 风格匹配
        → 召回更准
"""
import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ==================== HyDE Prompt ====================
HYDE_PROMPT = """请你扮演一位 AI/机器学习领域的研究员,根据下面的问题写一段简洁的答案。

要求:
1. 答案应该看起来像是从学术论文中摘录的段落
2. 包含相关的技术术语和细节(数学公式、模型名称、关键概念等)
3. 用陈述句而不是疑问句
4. 长度约 100-150 词
5. 即使你不确定具体细节,也要给出合理的猜测,不要说"我不知道"
6. 如果问题是中文,请用英文回答(因为知识库是英文论文)

问题: {question}

假设答案:"""


# ==================== HyDE 客户端 ====================
def get_hyde_client() -> OpenAI:
    """获取 LLM 客户端,用于生成假设答案"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未找到 DEEPSEEK_API_KEY")
    
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


def generate_hypothetical_answer(
    question: str,
    model: str = "deepseek-chat",
    temperature: float = 0.7,  # 稍高一些,鼓励生成多样化的"答案"
    max_tokens: int = 300,
) -> str:
    """
    用 LLM 生成假设答案
    
    Args:
        question: 用户原始查询
        model: LLM 模型
        temperature: 采样温度
        max_tokens: 最大生成长度
    
    Returns:
        LLM 生成的假设答案文本
    """
    client = get_hyde_client()
    prompt = HYDE_PROMPT.format(question=question)
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    return response.choices[0].message.content.strip()


def hyde_rewrite(
    question: str,
    include_original: bool = True,
) -> str:
    """
    HyDE 查询改写: 把原查询替换为假设答案
    
    Args:
        question: 原始查询
        include_original: 是否把原查询也拼接进去
                          (经验上拼接能稳定一些,纯假设答案有时会跑偏)
    
    Returns:
        改写后用于检索的查询文本
    """
    hypothetical = generate_hypothetical_answer(question)
    
    if include_original:
        # 经验技巧: 原查询 + 假设答案一起做检索
        # 这样既保留了原始意图,又用假设答案增强了风格匹配
        return f"{question}\n\n{hypothetical}"
    else:
        return hypothetical


# ==================== 测试入口 ====================
if __name__ == "__main__":
    test_queries = [
        "Explain the self-attention mechanism in Transformer",
        "什么是检索增强生成?它解决什么问题?",
        "LoRA 是什么? 它如何减少可训练参数?",
        "QLoRA 相对于 LoRA 的核心改进是什么?",
    ]
    
    for q in test_queries:
        print("=" * 70)
        print(f"🔍 原查询: {q}")
        print("=" * 70)
        
        hypothetical = generate_hypothetical_answer(q)
        print(f"\n💭 LLM 生成的假设答案:")
        print(hypothetical)
        print()
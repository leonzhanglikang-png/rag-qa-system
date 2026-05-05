"""
答案生成模块
基于检索到的上下文,调用 LLM 生成答案

设计要点:
- Prompt 工程: 明确指令防止幻觉、要求引用来源
- API 抽象: 支持切换不同 LLM(DeepSeek / Qwen / OpenAI)
- 流式输出: 后续 Gradio 集成时可改成流式以提升体验
"""
import os
from typing import List, Optional, Dict
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.documents import Document

load_dotenv()


# ==================== Prompt 模板 ====================
SYSTEM_PROMPT = """你是一个严谨的AI技术问答助手,专门基于提供的参考资料回答关于AI/机器学习论文的问题。

回答规则:
1. 严格基于"参考资料"中的内容回答,不要使用资料外的知识
2. 如果参考资料中没有足够信息,直接回答"根据提供的资料,我无法回答这个问题",不要编造
3. 回答时使用清晰的结构,必要时分点说明
4. 在回答末尾用 [来源: 文件名, 第X页] 的格式标注信息来源
5. 用户用什么语言提问,就用什么语言回答(中文问题用中文回答)
"""

USER_PROMPT_TEMPLATE = """参考资料:
{context}

---
问题: {question}

请基于上述参考资料回答问题。"""


# ==================== LLM 客户端 ====================
def get_llm_client() -> OpenAI:
    """获取 DeepSeek 客户端(API 兼容 OpenAI 格式)"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未找到 DEEPSEEK_API_KEY, 请检查 .env 文件")
    
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


# ==================== 上下文构造 ====================
def format_context(documents: List[Document]) -> str:
    """
    把检索到的 Document 列表格式化为 Prompt 中的 context 段落
    
    每个 chunk 标注序号和来源,方便 LLM 引用和用户溯源
    """
    if not documents:
        return "(无相关参考资料)"
    
    formatted = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source_file", "未知文件")
        page = doc.metadata.get("page", "?")
        content = doc.page_content.strip()
        
        formatted.append(
            f"【资料{i}】(来源: {source}, 第{page}页)\n{content}"
        )
    
    return "\n\n".join(formatted)


# ==================== 答案生成 ====================
def generate_answer(
    question: str,
    documents: List[Document],
    model: str = "deepseek-chat",
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> Dict[str, any]:
    """
    基于检索到的上下文生成答案
    
    Args:
        question: 用户问题
        documents: 检索到的相关 Document 列表
        model: LLM 模型名
        temperature: 采样温度,0.0-2.0,越低越确定
        max_tokens: 最大生成 token 数
    
    Returns:
        {
            "answer": 生成的答案,
            "sources": 引用的来源列表,
            "context": 构造的上下文(调试用),
        }
    """
    client = get_llm_client()
    context = format_context(documents)
    
    user_prompt = USER_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    answer = response.choices[0].message.content
    
    # 整理来源信息(去重)
    seen = set()
    sources = []
    for doc in documents:
        source = doc.metadata.get("source_file", "未知文件")
        page = doc.metadata.get("page", "?")
        key = (source, page)
        if key not in seen:
            seen.add(key)
            sources.append({"file": source, "page": page})
    
    return {
        "answer": answer,
        "sources": sources,
        "context": context,
    }


# ==================== 测试入口 ====================
if __name__ == "__main__":
    from loader import load_pdfs
    from splitter import split_documents
    from retriever import build_vectorstore, search
    
    # 1. 准备向量库(已存在则直接加载)
    print("=" * 60)
    print("准备向量库...")
    print("=" * 60)
    docs = load_pdfs()
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)
    print()
    
    # 2. 测试端到端 RAG: 检索 + 生成
    print("=" * 60)
    print("端到端 RAG 测试")
    print("=" * 60)
    
    test_questions = [
        "What is LoRA and how does it reduce trainable parameters?",
        "用中文解释 Transformer 的 self-attention 机制",
        "QLoRA 相对于 LoRA 的核心改进是什么?",
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"❓ 问题: {question}")
        print(f"{'='*60}")
        
        # 检索
        retrieved_docs = search(question, vectorstore, top_k=5)
        
        # 生成
        result = generate_answer(question, retrieved_docs)
        
        print(f"\n💡 回答:\n{result['answer']}")
        print(f"\n📚 引用来源:")
        for s in result["sources"]:
            print(f"   - {s['file']} (第{s['page']}页)")
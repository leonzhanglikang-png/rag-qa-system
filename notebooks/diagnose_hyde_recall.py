"""
深度诊断: HyDE 改写后的查询能不能找到 chunk 1461 (含核心公式)
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import sys
sys.path.insert(0, "src")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from hyde import generate_hypothetical_answer
import torch


# 准备向量库
device = "mps" if torch.backends.mps.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
)
vs = Chroma(
    collection_name="ai_papers",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)


def test_query(query: str, label: str):
    """跑纯向量检索, 看 attention.pdf 第3-4页的 chunk 在什么位置"""
    print(f"\n{'='*70}")
    print(f"🔍 [{label}] 查询: {query[:100]}{'...' if len(query)>100 else ''}")
    print(f"{'='*70}")
    
    # 拉 top-30, 看 attention 论文相关 chunk 排在哪
    results = vs.similarity_search_with_score(query, k=30)
    
    # 找出 attention.pdf 第 3-4 页的 chunks
    print("\n📊 attention.pdf 各页 chunks 在 top-30 中的排名:")
    for rank, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        if meta.get("source_file") == "attention_is_all_you_need.pdf":
            page = meta.get("page")
            chunk_id = meta.get("chunk_id")
            preview = doc.page_content[:80].replace("\n", " ")
            marker = "⭐" if page in [3, 4] else "  "
            print(f"  {marker} 排名 {rank:2d}: 第{page}页 chunk_id={chunk_id} | 分数={score:.4f}")
            print(f"        {preview}...")
    
    # 看看 top-5 都是什么
    print(f"\n📋 总 top-5:")
    for rank, (doc, score) in enumerate(results[:5], 1):
        meta = doc.metadata
        source = meta.get("source_file")
        page = meta.get("page")
        chunk_id = meta.get("chunk_id")
        print(f"  [{rank}] {source} 第{page}页 chunk_id={chunk_id} | 分数={score:.4f}")


if __name__ == "__main__":
    # 测试 1: 原始查询
    original = "Explain the self-attention mechanism in Transformer"
    test_query(original, "原始查询")
    
    # 测试 2: HyDE 改写后的查询
    print("\n💭 生成 HyDE 假设答案...")
    hypothetical = generate_hypothetical_answer(original)
    print(f"假设答案:\n{hypothetical}\n")
    
    # 注意: 用 hyde_rewrite 是 "原查询 + 假设答案"
    rewritten = f"{original}\n\n{hypothetical}"
    test_query(rewritten, "HyDE 改写 (原+假设)")
    
    # 测试 3: 只用假设答案(不拼原查询)
    test_query(hypothetical, "纯假设答案")
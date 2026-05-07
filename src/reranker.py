"""
Reranker 模块: cross-encoder 精排
模型: BAAI/bge-reranker-v2-m3 (中英双语,智源研究院)

设计意图:
- 召回阶段(BM25+向量): 高召回率,粗排序
- 精排阶段(本模块): 高精度,把最相关的拉到最前面
- Reranker 比 embedding 慢 5-10x, 但只对 top-20 做精排, 总耗时可控
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List
from langchain_core.documents import Document


# 全局单例(避免重复加载模型)
_reranker_instance = None


def get_reranker(model_name: str = "BAAI/bge-reranker-v2-m3"):
    """单例模式: reranker 模型只加载一次"""
    global _reranker_instance
    if _reranker_instance is None:
        from FlagEmbedding import FlagReranker
        print(f"🤖 加载 Reranker 模型: {model_name}")
        print(f"   (首次运行会下载约 568MB 模型, 后续直接用缓存)")
        _reranker_instance = FlagReranker(
            model_name,
            use_fp16=True,  # FP16 加速,Mac MPS 兼容
        )
        print("✅ Reranker 模型加载完成")
    return _reranker_instance


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int = 5,
) -> List[Document]:
    """
    用 cross-encoder 重排序候选文档
    
    Args:
        query: 用户查询
        documents: 候选 Document 列表(通常来自混合检索的 top-20)
        top_k: 精排后保留的 top-k
    
    Returns:
        按 reranker 分数降序排列的 top-k Document
    """
    if not documents:
        return []
    
    if len(documents) <= top_k:
        # 候选数已经少于 top_k, 不需要精排
        return documents
    
    reranker = get_reranker()
    
    # 构造 [query, doc_content] 对
    pairs = [[query, doc.page_content] for doc in documents]
    
    # 计算 reranker 分数
    # normalize=True 把分数归一化到 [0, 1], 越大越相关
    scores = reranker.compute_score(pairs, normalize=True)
    
    # 把分数和 doc 配对,按分数降序
    scored_docs = list(zip(scores, documents))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # 把 reranker 分数写入 metadata, 方便观察
    result = []
    for score, doc in scored_docs[:top_k]:
        doc.metadata["rerank_score"] = round(float(score), 4)
        result.append(doc)
    
    return result


# ==================== 测试入口 ====================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    
    from loader import load_pdfs
    from section_detector import enrich_documents_with_sections
    from splitter import split_documents
    from retriever import build_vectorstore, build_hybrid_retriever, search_hybrid
    
    # 准备数据
    print("=" * 70)
    print("准备 chunks + 向量库 + 混合检索器")
    print("=" * 70)
    docs = load_pdfs()
    docs = enrich_documents_with_sections(docs)
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    vectorstore = build_vectorstore(chunks)
    hybrid = build_hybrid_retriever(
        chunks, vectorstore,
        candidate_k=20,  # ⭐ 召回阶段拉 20 个候选, 给 reranker 精排空间
    )
    print()
    
    # 4 个核心测试查询
    test_queries = [
        ("Q1", "LoRA 是什么? 它如何减少可训练参数?"),
        ("Q2", "什么是检索增强生成?它解决什么问题?"),
        ("Q3", "Self-RAG 和普通 RAG 的区别是什么?"),
        ("Q4", "Explain the self-attention mechanism in Transformer"),
    ]
    
    for tag, query in test_queries:
        print("\n" + "=" * 70)
        print(f"🔍 {tag}: {query}")
        print("=" * 70)
        
        # Step 1: 混合检索召回 top-20
        candidates = search_hybrid(query, hybrid, top_k=20)
        print(f"\n📥 混合检索召回 {len(candidates)} 个候选")
        
        # 看一下召回的覆盖情况(有几个 lora/rag/...)
        from collections import Counter
        sources = Counter(d.metadata.get("source_file") for d in candidates)
        print(f"   来源分布: {dict(sources.most_common(3))}")
        
        # Step 2: Reranker 精排 top-5
        reranked = rerank_documents(query, candidates, top_k=5)
        
        print(f"\n[Reranker 精排后 top-5]")
        for i, doc in enumerate(reranked, 1):
            source = doc.metadata.get("source_file")
            page = doc.metadata.get("page")
            section = (doc.metadata.get("section", "?") or "?")[:30]
            score = doc.metadata.get("rerank_score")
            preview = doc.page_content[:100].replace("\n", " ")
            print(f"  [{i}] {source} 第{page}页 ({section}) | 分数={score}")
            print(f"      {preview}...")
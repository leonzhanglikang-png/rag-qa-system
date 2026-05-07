"""
对照实验: 向量检索 vs BM25 vs 混合检索
用同一组测试查询跑三种检索器,直接看效果差异
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import sys
sys.path.insert(0, "src")

from loader import load_pdfs
from section_detector import enrich_documents_with_sections
from splitter import split_documents
from retriever import build_vectorstore, search, build_hybrid_retriever, search_hybrid
from bm25_retriever import build_bm25_retriever, search_bm25


def print_results(results, label):
    """统一格式打印检索结果"""
    print(f"\n[{label}]")
    if not results:
        print("  (无结果)")
        return
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source_file", "?")
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section", "?")
        score = doc.metadata.get("retrieval_score", "")
        score_str = f" 分数={score}" if score else ""
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  [{i}] {source} 第{page}页 ({section[:30]}){score_str}")
        print(f"      {preview}...")


def main():
    # 准备数据
    print("=" * 70)
    print("准备 chunks + 向量库 + BM25 索引")
    print("=" * 70)
    docs = load_pdfs()
    docs = enrich_documents_with_sections(docs)
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    
    vectorstore = build_vectorstore(chunks)
    bm25 = build_bm25_retriever(chunks, k=5)
    hybrid = build_hybrid_retriever(
    chunks, vectorstore,
    bm25_weight=0.3, vector_weight=0.7,  # ← 给向量更多权重
    top_k=5, candidate_k=10
)
    print()
    
    # 测试查询(跟昨天 Day 1 / Day 2 用的一致, 方便横向对比)
    test_queries = [
        ("Q1: LoRA 是什么? 它如何减少可训练参数?", "lora.pdf 前几页"),
        ("Q2: 什么是检索增强生成?它解决什么问题?", "rag.pdf"),
        ("Q3: Self-RAG 和普通 RAG 的区别是什么?", "self_rag.pdf"),
        ("Q4: Explain the self-attention mechanism in Transformer", "attention.pdf"),
    ]
    
    for query, expected in test_queries:
        print("\n" + "=" * 70)
        print(f"🔍 {query}")
        print(f"📌 期望命中: {expected}")
        print("=" * 70)
        
        # 1. 纯向量检索
        vec_results = search(query.split(": ", 1)[1], vectorstore, top_k=5)
        print_results(vec_results, "向量检索")
        
        # 2. 纯 BM25
        bm25_results = search_bm25(query.split(": ", 1)[1], bm25, top_k=5)
        print_results(bm25_results, "BM25")
        
        # 3. 混合检索
        hybrid_results = search_hybrid(query.split(": ", 1)[1], hybrid, top_k=5)
        print_results(hybrid_results, "混合检索 (BM25 30% + 向量 70%)")


if __name__ == "__main__":
    main()
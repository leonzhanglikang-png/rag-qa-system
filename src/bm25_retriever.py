"""
BM25 检索器(关键词检索)
- 与向量检索互补
- 擅长精确关键词匹配, 不擅长语义理解
- 实现方式: LangChain 的 BM25Retriever (底层是 rank-bm25)
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import pickle
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


# 持久化路径(避免每次重新构建索引)
BM25_INDEX_PATH = "./bm25_index.pkl"


def build_bm25_retriever(
    chunks: List[Document],
    k: int = 5,
    persist_path: str = BM25_INDEX_PATH,
    force_rebuild: bool = False,
) -> BM25Retriever:
    """
    构建 BM25 检索器
    
    Args:
        chunks: 用于建索引的 Document 列表
        k: 返回 top-k 个结果
        persist_path: 持久化文件路径
        force_rebuild: 强制重建
    
    Returns:
        BM25Retriever 实例
    """
    # 注意: BM25Retriever 没有原生的 save/load 方法,
    # 我们手动 pickle 整个对象来实现持久化
    if Path(persist_path).exists() and not force_rebuild:
        print(f"📂 加载已存在的 BM25 索引: {persist_path}")
        with open(persist_path, "rb") as f:
            retriever = pickle.load(f)
        retriever.k = k
        return retriever
    
    print(f"🔨 构建 BM25 索引, 共 {len(chunks)} 个 chunk...")
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    
    # 持久化
    with open(persist_path, "wb") as f:
        pickle.dump(retriever, f)
    print(f"✅ BM25 索引已保存: {persist_path}")
    
    return retriever


def search_bm25(
    query: str,
    retriever: BM25Retriever,
    top_k: Optional[int] = None,
) -> List[Document]:
    """用 BM25 检索"""
    if top_k is not None:
        retriever.k = top_k
    return retriever.invoke(query)


# ==================== 测试入口 ====================
if __name__ == "__main__":
    from loader import load_pdfs
    from section_detector import enrich_documents_with_sections
    from splitter import split_documents
    
    # 复用现有 chunks(同一份数据,确保跟向量库的对照公平)
    print("=" * 70)
    print("Step 1.1: 加载 chunks (与向量库使用相同数据)")
    print("=" * 70)
    docs = load_pdfs()
    docs = enrich_documents_with_sections(docs)
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    print(f"📦 共 {len(chunks)} 个 chunk\n")
    
    # 构建 BM25
    print("=" * 70)
    print("Step 1.2: 构建 BM25 索引")
    print("=" * 70)
    bm25 = build_bm25_retriever(chunks, k=5, force_rebuild=True)
    print()
    
    # ============================================================
    # 关键测试: 拿昨天的 Q1 来测 BM25, 看它能不能召回 lora.pdf 第1页
    # ============================================================
    print("=" * 70)
    print("Step 1.3: BM25 vs 向量检索对照实验")
    print("=" * 70)
    
    test_queries = [
        # 昨天的 Q1, 向量检索召回不了第1页
        "LoRA reduce trainable parameters",
        # 昨天的 Q3, 向量检索把qlora数据集排第一(bad case)
        "什么是检索增强生成?它解决什么问题?",
        # 跨语言测试: BM25 应该完全不行(中文查英文知识库)
        "Transformer 的 self-attention 机制",
        # 数字/术语测试: BM25 应该擅长
        "QLoRA 4-bit NormalFloat",
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"🔍 查询: {query}")
        print(f"{'='*70}")
        
        bm25_results = search_bm25(query, bm25, top_k=5)
        
        print("\n[BM25 召回 top-5]")
        for i, doc in enumerate(bm25_results, 1):
            source = doc.metadata.get("source_file")
            page = doc.metadata.get("page")
            section = doc.metadata.get("section", "?")
            preview = doc.page_content[:100].replace("\n", " ")
            print(f"  [{i}] {source} 第{page}页 ({section})")
            print(f"      {preview}...")
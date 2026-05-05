"""
检索模块
负责把切片后的文档向量化、存入向量库,并提供语义检索能力

技术栈:
- Embedding: BAAI/bge-small-zh-v1.5 (中文场景经典选择, 512维)
- 向量库: ChromaDB (轻量级, 适合中小规模知识库)
- 加速: Apple Silicon MPS (Mac M系列芯片GPU加速)
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from typing import List, Optional
from pathlib import Path


import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ==================== 全局配置 ====================
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
PERSIST_DIR = "./chroma_db"  # 向量库持久化目录
COLLECTION_NAME = "ai_papers"  # 集合名(类似数据库的"表名")


def get_device() -> str:
    """自动选择最佳计算设备: MPS(Mac GPU) > CUDA(N卡) > CPU"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    加载 embedding 模型
    
    bge-small-zh-v1.5 说明:
    - 智源研究院开源的中文 embedding 模型
    - 输出 512 维向量
    - 在 C-MTEB 中文 benchmark 上效果稳定
    - 模型大小约 100MB, CPU/Mac 都能跑
    """
    device = get_device()
    print(f"🖥️  使用设备: {device}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,  # L2 归一化, 让点积等价于余弦相似度
            "batch_size": 32,              # 批处理加速
        },
    )
    return embeddings


def build_vectorstore(
    chunks: List[Document],
    persist_dir: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    force_rebuild: bool = False,
) -> Chroma:
    """
    构建(或加载)向量库
    
    Args:
        chunks: 切片后的 Document 列表
        persist_dir: 向量库持久化目录
        collection_name: 集合名
        force_rebuild: True 则强制重建(删除旧库)
    
    Returns:
        Chroma 向量库对象
    """
    embeddings = get_embeddings()
    
    # 如果已经存在向量库且不强制重建, 直接加载
    if os.path.exists(persist_dir) and not force_rebuild:
        print(f"📂 检测到已有向量库, 直接加载: {persist_dir}")
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        # 验证是否真的有数据
        count = vectorstore._collection.count()
        if count > 0:
            print(f"✅ 加载成功, 向量库中已有 {count} 条记录")
            return vectorstore
        else:
            print("⚠️  向量库为空, 需要重新构建")
    
    # 构建新向量库
    print(f"🔨 开始构建向量库, 共 {len(chunks)} 个 chunk 待向量化...")
    print(f"   (首次运行会下载 embedding 模型, 约 100MB, 请耐心等待)")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    
    count = vectorstore._collection.count()
    print(f"✅ 向量库构建完成, 共 {count} 条记录, 持久化到 {persist_dir}")
    return vectorstore


def search(
    query: str,
    vectorstore: Chroma,
    top_k: int = 5,
    score_threshold: Optional[float] = None,
) -> List[Document]:
    """
    语义检索: 给定查询, 返回最相关的 chunks
    
    Args:
        query: 用户问题
        vectorstore: Chroma 向量库
        top_k: 返回最相关的前几个
        score_threshold: 相似度阈值(None表示不过滤)
    
    Returns:
        相关 Document 列表(按相关度排序)
    """
    # 带分数的检索, 方便我们看每个结果的相似度
    results_with_scores = vectorstore.similarity_search_with_score(query, k=top_k)
    
    # 可选: 按阈值过滤
    if score_threshold is not None:
        results_with_scores = [
            (doc, score) for doc, score in results_with_scores
            if score >= score_threshold
        ]
    
    # 把分数也塞进 metadata 方便后续展示
    docs = []
    for doc, score in results_with_scores:
        doc.metadata["retrieval_score"] = round(float(score), 4)
        docs.append(doc)
    
    return docs


# ==================== 测试入口 ====================
if __name__ == "__main__":
    from loader import load_pdfs
    from splitter import split_documents
    
    # 1. 加载 + 切片
    print("=" * 60)
    print("Step 1: 加载 PDF 并切片")
    print("=" * 60)
    docs = load_pdfs()
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    print(f"📦 切片完成: {len(chunks)} 个 chunk\n")
    
    # 2. 构建向量库
    print("=" * 60)
    print("Step 2: 构建向量库 (首次运行会下载模型并向量化, 约3-8分钟)")
    print("=" * 60)
    vectorstore = build_vectorstore(chunks)
    print()
    
    # 3. 测试检索: 跑几个真实问题
    print("=" * 60)
    print("Step 3: 测试语义检索")
    print("=" * 60)
    
    test_queries = [
        "What is LoRA and how does it reduce trainable parameters?",
        "Explain the self-attention mechanism in Transformer",
        "什么是检索增强生成?它解决什么问题?",
        "QLoRA 相比 LoRA 有什么改进?",
    ]
    
    for query in test_queries:
        print(f"\n🔍 查询: {query}")
        print("-" * 60)
        results = search(query, vectorstore, top_k=3)
        
        for i, doc in enumerate(results, 1):
            score = doc.metadata.get("retrieval_score")
            source = doc.metadata.get("source_file")
            page = doc.metadata.get("page")
            preview = doc.page_content[:120].replace("\n", " ")
            print(f"  [{i}] 来源: {source} | 第{page}页 | 相似度分数: {score}")
            print(f"      {preview}...")
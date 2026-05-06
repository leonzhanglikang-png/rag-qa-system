"""
语义切片模块
基于 embedding 相似度的智能切片,与递归切片做 A/B 对比

核心思想:
- 递归切片: 按字符数+标点强切, 可能切断语义
- 语义切片: 按句子语义跳跃点切, 保留语义完整性
"""
from typing import List, Literal
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import torch


def get_embeddings_for_chunking() -> HuggingFaceEmbeddings:
    """复用 retriever.py 里的 embedding 配置"""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )


def split_documents_semantic(
    documents: List[Document],
    breakpoint_threshold_type: Literal[
        "percentile", "standard_deviation", "interquartile", "gradient"
    ] = "percentile",
    breakpoint_threshold_amount: float = 95.0,
) -> List[Document]:
    """
    语义切片: 在 embedding 相似度突变点切分文档
    
    Args:
        documents: 输入的 Document 列表
        breakpoint_threshold_type: 切点判定方法
            - percentile: 取相似度差异分布的第 N 百分位作为切点(最常用)
            - standard_deviation: 用 N 倍标准差判定
            - interquartile: 用四分位距判定
            - gradient: 用梯度方法
        breakpoint_threshold_amount: 阈值的具体数值
            - percentile=95 表示: 只有当相邻句子相似度差异超过 95% 的差异时才切
            - 越小切得越碎,越大 chunk 越大
    
    Returns:
        切片后的 Document 列表
    """
    embeddings = get_embeddings_for_chunking()
    
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )
    
    chunks = splitter.split_documents(documents)
    
    # 过滤过短 chunk(同 splitter.py)
    MIN_CHUNK_LENGTH = 50
    before = len(chunks)
    chunks = [c for c in chunks if len(c.page_content) >= MIN_CHUNK_LENGTH]
    if before > len(chunks):
        print(f"  🧹 过滤短 chunk: {before} → {len(chunks)}")
    
    # 注入 chunk_id
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    
    return chunks


def analyze_chunks(chunks: List[Document]) -> dict:
    """复用 splitter.py 的统计函数"""
    if not chunks:
        return {}
    lengths = [len(c.page_content) for c in chunks]
    
    # 计算长度方差(评估切片均匀度的指标)
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    std_dev = variance ** 0.5
    
    return {
        "总数": len(chunks),
        "平均长度": round(mean, 1),
        "标准差": round(std_dev, 1),
        "最短": min(lengths),
        "最长": max(lengths),
        "覆盖文档": len(set(c.metadata.get("source_file") for c in chunks)),
    }


if __name__ == "__main__":
    import time
    from loader import load_pdfs
    from splitter import split_documents
    
    docs = load_pdfs()
    print()
    
    # 实验 1: 递归切片(基线,你昨天用的)
    print("=" * 70)
    print("实验1: RecursiveCharacterTextSplitter (基线)")
    print("=" * 70)
    t1 = time.time()
    recursive_chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    t1_elapsed = time.time() - t1
    
    stats1 = analyze_chunks(recursive_chunks)
    stats1["耗时(秒)"] = round(t1_elapsed, 2)
    for k, v in stats1.items():
        print(f"  {k}: {v}")
    
    # 实验 2: 语义切片(新方法)
    print("\n" + "=" * 70)
    print("实验2: SemanticChunker (percentile=95)")
    print("⏳ 此步会用 embedding 模型计算相似度, 预计 3-5 分钟...")
    print("=" * 70)
    t2 = time.time()
    semantic_chunks = split_documents_semantic(
        docs,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95.0,
    )
    t2_elapsed = time.time() - t2
    
    stats2 = analyze_chunks(semantic_chunks)
    stats2["耗时(秒)"] = round(t2_elapsed, 2)
    for k, v in stats2.items():
        print(f"  {k}: {v}")
    
    # 对比
    print("\n" + "=" * 70)
    print("📊 对比小结")
    print("=" * 70)
    print(f"{'指标':<15} | {'递归切片':<15} | {'语义切片':<15} | {'差异'}")
    print("-" * 70)
    for key in ["总数", "平均长度", "标准差", "最短", "最长", "耗时(秒)"]:
        v1 = stats1.get(key, "-")
        v2 = stats2.get(key, "-")
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            diff = f"{v2 - v1:+.1f}"
        else:
            diff = "-"
        print(f"{key:<15} | {str(v1):<15} | {str(v2):<15} | {diff}")
    
    # 看一个具体语义 chunk
    print("\n" + "=" * 70)
    print("语义切片的第 5 个 chunk 示例:")
    print("=" * 70)
    if len(semantic_chunks) > 5:
        print(f"长度: {len(semantic_chunks[5].page_content)}")
        print(f"内容前 600 字符:")
        print(semantic_chunks[5].page_content[:600])
        print(f"\n元数据: {semantic_chunks[5].metadata}")
        
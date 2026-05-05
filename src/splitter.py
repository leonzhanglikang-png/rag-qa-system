"""
文本切片模块
将长文档切分为适合检索的小块(chunk)

核心策略: RecursiveCharacterTextSplitter 递归切片
- 优先按段落分割,段落太长再按句子,句子太长再按字符
- 比简单的固定长度切片保留更多语义完整性
"""
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    strategy: str = "recursive",
) -> List[Document]:
    """
    把 Document 列表切分成更小的 chunk

    Args:
        documents: 输入的 Document 列表(每个是 PDF 的一页)
        chunk_size: 每个 chunk 的最大字符数
        chunk_overlap: 相邻 chunk 之间的重叠字符数(防止语义在边界被切断)
        strategy: "recursive"(推荐) 或 "fixed"(对照实验用)

    Returns:
        切片后的 Document 列表
    """
    if strategy == "recursive":
        # 递归切片: 按优先级尝试不同分隔符
        # 中英文混合,所以分隔符既有英文标点也有中文标点
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",            # 优先级1: 段落
                "\n",              # 优先级2: 换行
                "。", "！", "？",    # 优先级3: 中文句号
                ". ", "! ", "? ",  # 优先级4: 英文句号
                "，", ", ",         # 优先级5: 逗号
                " ",                # 优先级6: 空格(单词边界)
                "",                 # 优先级7: 字符级兜底
            ],
        )
    elif strategy == "fixed":
        # 固定长度切片: 简单粗暴,作为对照
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="",  # 不按分隔符,纯按长度切
        )
    else:
        raise ValueError(f"未知策略: {strategy}")

    chunks = splitter.split_documents(documents)
    # 过滤掉过短的 chunk(PDF 解析残留, 如孤立页码、单字符)
    MIN_CHUNK_LENGTH = 50
    before_filter = len(chunks)
    chunks = [c for c in chunks if len(c.page_content) >= MIN_CHUNK_LENGTH]
    after_filter = len(chunks)
    if before_filter > after_filter:
        print(f"  🧹 过滤短 chunk: {before_filter} → {after_filter} (移除 {before_filter - after_filter} 个)")
    
    # 给每个 chunk 加一个唯一 ID,后续检索/评估会用到
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


def analyze_chunks(chunks: List[Document]) -> dict:
    """统计 chunk 的基本信息,用于面试讲项目时的具体数字"""
    if not chunks:
        return {}

    lengths = [len(c.page_content) for c in chunks]
    return {
        "总数": len(chunks),
        "平均长度": round(sum(lengths) / len(lengths), 1),
        "最短": min(lengths),
        "最长": max(lengths),
        "覆盖文档": len(set(c.metadata.get("source_file") for c in chunks)),
    }


if __name__ == "__main__":
    from loader import load_pdfs

    docs = load_pdfs()

    # 实验1: 递归切片(默认推荐)
    print("\n" + "=" * 60)
    print("实验1: RecursiveCharacterTextSplitter (chunk=500, overlap=50)")
    print("=" * 60)
    chunks_recursive = split_documents(docs, chunk_size=500, chunk_overlap=50)
    stats1 = analyze_chunks(chunks_recursive)
    for k, v in stats1.items():
        print(f"  {k}: {v}")

    # 实验2: 固定切片(对照)
    print("\n" + "=" * 60)
    print("实验2: CharacterTextSplitter (chunk=500, overlap=50)")
    print("=" * 60)
    chunks_fixed = split_documents(
        docs, chunk_size=500, chunk_overlap=50, strategy="fixed"
    )
    stats2 = analyze_chunks(chunks_fixed)
    for k, v in stats2.items():
        print(f"  {k}: {v}")

    # 看一个具体 chunk 长啥样
    print("\n" + "=" * 60)
    print("递归切片的第一个 chunk 示例:")
    print("=" * 60)
    print(f"内容长度: {len(chunks_recursive[0].page_content)}")
    print(f"内容: {chunks_recursive[0].page_content}")
    print(f"\n元数据: {chunks_recursive[0].metadata}")
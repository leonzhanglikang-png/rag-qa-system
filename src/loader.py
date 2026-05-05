"""
PDF文档加载模块
负责把data/raw_pdfs/下的所有PDF读入内存,转成LangChain的Document对象
"""
import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdfs(pdf_dir: str = "data/raw_pdfs") -> List[Document]:
    """
    加载指定目录下所有PDF文件
    
    Args:
        pdf_dir: PDF文件夹路径
    
    Returns:
        Document对象列表,每个Document对应PDF的一页
    """
    pdf_dir = Path(pdf_dir)
    
    if not pdf_dir.exists():
        raise FileNotFoundError(f"目录不存在: {pdf_dir}")
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"目录 {pdf_dir} 下没有PDF文件")
    
    print(f"📚 找到 {len(pdf_files)} 个PDF文件,开始加载...")
    
    all_docs = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            
            # 给每个文档加上来源元数据(后面检索时能溯源)
            for doc in docs:
                doc.metadata["source_file"] = pdf_path.name
            
            all_docs.extend(docs)
            print(f"  ✅ {pdf_path.name}: {len(docs)} 页")
        except Exception as e:
            print(f"  ❌ {pdf_path.name} 加载失败: {e}")
    
    print(f"\n📊 总计加载 {len(all_docs)} 页文档")
    return all_docs


if __name__ == "__main__":
    # 直接运行这个文件就能测试
    docs = load_pdfs()
    
    # 打印第一页内容预览,验证加载成功
    if docs:
        print("\n" + "="*60)
        print("第一页内容预览:")
        print("="*60)
        print(docs[0].page_content[:500])
        print("\n元数据:", docs[0].metadata)
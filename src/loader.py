"""
PDF 文档加载模块 v2
策略: PyPDF 提取正文(可读性好) + pdfplumber 单独提取表格(结构化)

升级说明 (vs v1):
- v1: 仅用 PyPDFLoader, 表格内容丢失, 学术论文双栏排版有时混乱
- v2: 双解析器协同, 正文走 PyPDF, 表格走 pdfplumber, 兼顾可读性和结构化
"""
import os
from pathlib import Path
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import pdfplumber


def _extract_tables_per_page(pdf_path: str) -> Dict[int, List[str]]:
    """
    用 pdfplumber 单独提取每页的表格,转成 markdown 格式
    
    Returns:
        {page_index: [markdown_table_str, ...]}, page_index 从 0 开始
    """
    tables_by_page = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if not tables:
                    continue
                
                md_tables = []
                for tbl in tables:
                    if not tbl or len(tbl) < 2:
                        continue  # 至少要有表头+一行数据
                    
                    md_lines = []
                    # 表头
                    header = [str(c or "").strip() for c in tbl[0]]
                    md_lines.append("| " + " | ".join(header) + " |")
                    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
                    # 数据行
                    for row in tbl[1:]:
                        cells = [str(c or "").strip() for c in row]
                        md_lines.append("| " + " | ".join(cells) + " |")
                    
                    md_tables.append("\n".join(md_lines))
                
                if md_tables:
                    tables_by_page[page_idx] = md_tables
    except Exception as e:
        print(f"  ⚠️  pdfplumber 表格提取失败: {e}")
    
    return tables_by_page


def _load_single_pdf(pdf_path: Path) -> List[Document]:
    """
    加载单个 PDF: PyPDF 提取正文 + pdfplumber 补充表格
    """
    # Step 1: 用 PyPDF 提取正文(可读性好)
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    
    # Step 2: 用 pdfplumber 单独提取表格
    tables_by_page = _extract_tables_per_page(str(pdf_path))
    
    # Step 3: 把表格拼接到对应页面的文本末尾
    for doc in docs:
        page_idx = doc.metadata.get("page", 0)
        doc.metadata["source_file"] = pdf_path.name
        doc.metadata["has_tables"] = False  # 默认无表
        
        if page_idx in tables_by_page:
            tables = tables_by_page[page_idx]
            table_section = "\n\n[本页包含 {} 个表格]\n\n".format(len(tables))
            table_section += "\n\n".join(tables)
            doc.page_content += table_section
            doc.metadata["has_tables"] = True
            doc.metadata["num_tables"] = len(tables)
    
    return docs


def load_pdfs(pdf_dir: str = "data/raw_pdfs") -> List[Document]:
    """
    加载指定目录下所有 PDF 文件 (v2: 双解析器协同)
    """
    pdf_dir = Path(pdf_dir)
    
    if not pdf_dir.exists():
        raise FileNotFoundError(f"目录不存在: {pdf_dir}")
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"目录 {pdf_dir} 下没有 PDF 文件")
    
    print(f"📚 找到 {len(pdf_files)} 个 PDF 文件,开始加载 (v2: PyPDF+pdfplumber)...")
    
    all_docs = []
    total_tables = 0
    for pdf_path in pdf_files:
        try:
            docs = _load_single_pdf(pdf_path)
            tables_in_file = sum(
                d.metadata.get("num_tables", 0) for d in docs
            )
            total_tables += tables_in_file
            
            tag = f" (含 {tables_in_file} 个表格)" if tables_in_file > 0 else ""
            print(f"  ✅ {pdf_path.name}: {len(docs)} 页{tag}")
            all_docs.extend(docs)
        except Exception as e:
            print(f"  ❌ {pdf_path.name} 加载失败: {e}")
    
    print(f"\n📊 总计: {len(all_docs)} 页文档, {total_tables} 个表格")
    return all_docs


if __name__ == "__main__":
    docs = load_pdfs()
    
    if docs:
        # 找一个含表格的页面看效果
        table_docs = [d for d in docs if d.metadata.get("has_tables")]
        
        print("\n" + "=" * 60)
        print("第一页内容预览:")
        print("=" * 60)
        print(docs[0].page_content[:500])
        print("\n元数据:", docs[0].metadata)
        
        if table_docs:
            print("\n" + "=" * 60)
            print(f"含表格的页面示例 (来自 {table_docs[0].metadata['source_file']} 第{table_docs[0].metadata['page']}页):")
            print("=" * 60)
            print(table_docs[0].page_content[-800:])  # 看末尾,因为表格拼在最后
            print("\n元数据:", table_docs[0].metadata)
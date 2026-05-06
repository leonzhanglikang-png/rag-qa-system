"""
Day 2 - Step 1.2: PDF 解析器对比实验

目的: 同一份 PDF 用 3 种解析器分别跑,对比:
  1. 提取速度
  2. 总字符数(覆盖度)
  3. 表格识别能力
  4. 排版还原度(看输出文本是否有序)

结论用于决策 loader.py 升级方向
"""
import time
from pathlib import Path

# 测试用的 PDF: LoRA 论文(含超参数表格 + 数学公式 + 双栏排版)
TEST_PDF = "data/raw_pdfs/lora.pdf"


def test_pypdf():
    """方法1: PyPDFLoader (LangChain 默认, 你昨天用的)"""
    from langchain_community.document_loaders import PyPDFLoader
    
    start = time.time()
    loader = PyPDFLoader(TEST_PDF)
    docs = loader.load()
    elapsed = time.time() - start
    
    total_chars = sum(len(d.page_content) for d in docs)
    return {
        "解析器": "PyPDF",
        "页数": len(docs),
        "总字符数": total_chars,
        "耗时(秒)": round(elapsed, 2),
        "样本(第3页前300字符)": docs[2].page_content[:300] if len(docs) > 2 else "",
    }


def test_pdfplumber():
    """方法2: pdfplumber (表格之王)"""
    import pdfplumber
    
    start = time.time()
    pages_text = []
    with pdfplumber.open(TEST_PDF) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
    elapsed = time.time() - start
    
    total_chars = sum(len(t) for t in pages_text)
    return {
        "解析器": "pdfplumber",
        "页数": len(pages_text),
        "总字符数": total_chars,
        "耗时(秒)": round(elapsed, 2),
        "样本(第3页前300字符)": pages_text[2][:300] if len(pages_text) > 2 else "",
    }


def test_pdfplumber_with_tables():
    """方法3: pdfplumber + 表格独立提取(进阶,展示能力)"""
    import pdfplumber
    
    start = time.time()
    pages_text = []
    total_tables = 0
    with pdfplumber.open(TEST_PDF) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            tables = page.extract_tables()
            total_tables += len(tables)
            
            # 把识别到的表格转成 markdown 格式拼到文本后
            if tables:
                for tbl in tables:
                    md_table = "\n[TABLE]\n"
                    for row in tbl:
                        md_table += " | ".join(str(cell or "") for cell in row) + "\n"
                    md_table += "[/TABLE]\n"
                    text += md_table
            
            pages_text.append(text)
    elapsed = time.time() - start
    
    total_chars = sum(len(t) for t in pages_text)
    return {
        "解析器": "pdfplumber + tables",
        "页数": len(pages_text),
        "总字符数": total_chars,
        "识别到的表格数": total_tables,
        "耗时(秒)": round(elapsed, 2),
        "样本(第3页前300字符)": pages_text[2][:300] if len(pages_text) > 2 else "",
    }


def main():
    print("=" * 70)
    print(f"📄 测试 PDF: {TEST_PDF}")
    print("=" * 70)
    
    if not Path(TEST_PDF).exists():
        print(f"❌ 找不到 {TEST_PDF}")
        return
    
    results = []
    
    print("\n[1/3] 测试 PyPDF...")
    results.append(test_pypdf())
    
    print("[2/3] 测试 pdfplumber...")
    results.append(test_pdfplumber())
    
    print("[3/3] 测试 pdfplumber + 表格识别...")
    results.append(test_pdfplumber_with_tables())
    
    # 汇总对比
    print("\n" + "=" * 70)
    print("📊 对比结果")
    print("=" * 70)
    for r in results:
        print(f"\n【{r['解析器']}】")
        for k, v in r.items():
            if k == "解析器":
                continue
            if k == "样本(第3页前300字符)":
                print(f"  {k}:")
                print(f"  {'-' * 60}")
                print("  " + v.replace("\n", "\n  "))
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
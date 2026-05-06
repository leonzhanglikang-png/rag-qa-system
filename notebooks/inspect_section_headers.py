"""
Day 2 - Step 3.1: 调研论文里章节标题的真实格式

目的: 在写章节识别规则前, 先扫一遍真实数据,
看每页开头几行长什么样, 据此设计识别规则
"""
import re
from pathlib import Path
import pdfplumber


def get_first_lines_per_page(pdf_path: str, n_lines: int = 3):
    """提取每页开头的 n 行"""
    pages_first_lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            pages_first_lines.append(lines[:n_lines])
    return pages_first_lines


def main():
    pdf_dir = Path("data/raw_pdfs")
    
    # 只扫 3 篇代表性论文 (避免输出过长)
    sample_pdfs = ["lora.pdf", "rag.pdf", "attention_is_all_you_need.pdf"]
    
    for pdf_name in sample_pdfs:
        pdf_path = pdf_dir / pdf_name
        if not pdf_path.exists():
            continue
        
        print("=" * 70)
        print(f"📄 {pdf_name}")
        print("=" * 70)
        
        first_lines = get_first_lines_per_page(str(pdf_path))
        
        for page_idx, lines in enumerate(first_lines):
            print(f"\n--- 第 {page_idx} 页开头 ---")
            for line in lines:
                # 截断显示, 避免单行过长
                display = line[:80] + "..." if len(line) > 80 else line
                print(f"  | {display}")


if __name__ == "__main__":
    main()
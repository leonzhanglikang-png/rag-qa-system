"""
章节标题识别模块 v2 (rule-based + 多重过滤)
"""
import re
from typing import List, Dict, Optional
from collections import Counter
from langchain_core.documents import Document


# ==================== 规则配置 ====================
COMMON_SECTIONS = {
    "abstract", "introduction", "background", "related work", "preliminaries",
    "method", "methodology", "approach", "model", "models", "framework", "architecture",
    "experiment", "experiments", "experimental setup", "implementation",
    "result", "results", "evaluation", "analysis", "discussion",
    "conclusion", "conclusions", "limitation", "limitations",
    "future work", "broader impact", "broader impacts", "ethical concerns",
    "references", "bibliography", "appendix", "acknowledgment", "acknowledgments",
}

# 已知误识别黑名单(模型名/benchmark/数据集名)
BLACKLIST = {
    "bert", "gpt", "rag", "cnn", "rnn", "lstm", "gru",
    "bleu", "rouge", "lambada", "squad", "glue", "superglue",
    "imagenet", "wikipedia", "wsj",
    "qa", "nlp", "ml", "ai",
}


# ==================== 工具函数 ====================
def _is_likely_title(line: str) -> bool:
    """启发式排除论文大标题"""
    if line.isupper() and len(line) > 50:
        return True
    if any(s in line for s in [":", "—", "–"]) and len(line) > 40:
        return True
    return False


def _has_repeated_words(line: str) -> bool:
    """检查是否有重复词(如 'Bert Bert', 'Csqa Csqa')"""
    words = line.lower().split()
    if len(words) < 2:
        return False
    counter = Counter(words)
    return any(count >= 2 for count in counter.values())


def _has_too_many_short_tokens(line: str) -> bool:
    """检查是否充斥孤立单字符(如 'S A U S A U S A')"""
    tokens = line.split()
    if not tokens:
        return False
    short_tokens = [t for t in tokens if len(t) <= 2]
    return len(short_tokens) / len(tokens) > 0.5


def _is_blacklisted(line: str) -> bool:
    """检查是否在黑名单中"""
    cleaned = re.sub(r"[^\w\s]", "", line).lower().strip()
    return cleaned in BLACKLIST


# ==================== 核心识别 ====================
def detect_section_in_line(line: str) -> Optional[Dict]:
    """判断一行文本是否是章节标题"""
    line = line.strip()
    if not line or len(line) < 3 or len(line) > 80:
        return None
    
    # 多重过滤(任一命中就排除)
    if _is_likely_title(line):
        return None
    if _has_repeated_words(line):
        return None
    if _has_too_many_short_tokens(line):
        return None
    if _is_blacklisted(line):
        return None
    
    # 规则 1 (高置信度): 数字编号 + 标题
    m = re.match(r"^(\d+(?:\.\d+)*)\s+([A-Za-z][A-Za-z\s\-]{2,60})$", line)
    if m:
        number, title = m.group(1), m.group(2).strip()
        # 黑名单二次检查
        if title.lower().strip() in BLACKLIST:
            return None
        level = number.count(".") + 1
        return {
            "section": f"{number} {title}",
            "level": level,
            "confidence": "high",
        }
    
    # 规则 2 (中置信度): 字母编号 + 标题(附录)
    m = re.match(r"^([A-Z](?:\.\d+)*)\s+([A-Z][A-Za-z\s\-]{4,60})$", line)
    if m:
        letter, title = m.group(1), m.group(2).strip()
        # 排除 'A B C D' 这种连续单字母列
        if len(title) < 5:
            return None
        return {
            "section": f"Appendix {letter}: {title}",
            "level": 1,
            "confidence": "medium",
        }
    
    # 规则 3 (高置信度): 标准章节关键词单独成行
    if line.lower() in COMMON_SECTIONS:
        return {
            "section": line.title(),
            "level": 1,
            "confidence": "high",
        }
    
    # 规则 4 (中置信度): 全大写短标题
    # 收紧: 3-35 字符,2-5 个词,不含数字
    if (
        line.isupper()
        and 3 <= len(line) <= 35
        and 2 <= len(line.split()) <= 5
        and not any(c.isdigit() for c in line)
        and re.match(r"^[A-Z][A-Z\s\-]+$", line)
    ):
        return {
            "section": line.title(),
            "level": 1,
            "confidence": "medium",
        }
    
    return None


def detect_section_for_page(page_text: str) -> Optional[Dict]:
    """
    扫一页文本,找出第一个像章节标题的行
    扩大到前 10 行,提高覆盖率
    """
    lines = [l.strip() for l in page_text.split("\n") if l.strip()]
    
    for line in lines[:10]:
        result = detect_section_in_line(line)
        if result:
            return result
    
    return None


def enrich_documents_with_sections(documents: List[Document]) -> List[Document]:
    """给 Document 列表注入 section 元数据"""
    by_file = {}
    for doc in documents:
        fname = doc.metadata.get("source_file", "unknown")
        by_file.setdefault(fname, []).append(doc)
    
    total_detected = 0
    total_inherited = 0
    
    for fname, file_docs in by_file.items():
        file_docs.sort(key=lambda d: d.metadata.get("page", 0))
        
        current_section = "Title/Abstract"
        current_confidence = "rule"
        
        for doc in file_docs:
            page_idx = doc.metadata.get("page", 0)
            
            if page_idx == 0:
                doc.metadata["section"] = "Title/Abstract"
                doc.metadata["section_confidence"] = "rule"
                continue
            
            detected = detect_section_for_page(doc.page_content)
            
            if detected:
                current_section = detected["section"]
                current_confidence = detected["confidence"]
                doc.metadata["section"] = current_section
                doc.metadata["section_confidence"] = current_confidence
                total_detected += 1
            else:
                doc.metadata["section"] = current_section
                doc.metadata["section_confidence"] = "inherited"
                total_inherited += 1
    
    print(f"  📑 章节识别: {total_detected} 页直接识别, {total_inherited} 页继承上页")
    return documents


# ==================== 测试入口(同 v1) ====================
if __name__ == "__main__":
    from loader import load_pdfs
    
    docs = load_pdfs()
    print()
    
    print("=" * 70)
    print("注入章节元数据 (v2: 加强过滤)...")
    print("=" * 70)
    docs = enrich_documents_with_sections(docs)
    
    print("\n" + "=" * 70)
    print("📊 各论文章节识别情况")
    print("=" * 70)
    
    by_file = {}
    for d in docs:
        fname = d.metadata.get("source_file")
        section = d.metadata.get("section", "Unknown")
        confidence = d.metadata.get("section_confidence", "?")
        by_file.setdefault(fname, []).append((d.metadata.get("page"), section, confidence))
    
    for fname in sorted(by_file.keys()):
        print(f"\n📄 {fname}")
        sections_in_file = Counter()
        for page, sec, conf in by_file[fname]:
            sections_in_file[sec] += 1
        
        for sec, count in sections_in_file.most_common():
            print(f"   {sec}: {count} 页")
    
    print("\n" + "=" * 70)
    print("📊 整体识别置信度分布")
    print("=" * 70)
    confidence_counter = Counter(d.metadata.get("section_confidence") for d in docs)
    for conf, count in confidence_counter.most_common():
        pct = count / len(docs) * 100
        print(f"  {conf}: {count} 页 ({pct:.1f}%)")
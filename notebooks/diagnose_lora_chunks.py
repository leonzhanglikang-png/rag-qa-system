"""
诊断: 看看 LoRA 论文的 chunk 在向量库里是怎么分布的
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from collections import Counter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch


device = "mps" if torch.backends.mps.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
)

vs = Chroma(
    collection_name="ai_papers",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# 拿出所有 chunk 的元数据
all_data = vs.get(include=["metadatas"])
metadatas = all_data["metadatas"]

# 按文件统计
by_file = Counter(m.get("source_file") for m in metadatas)
print("📊 各 PDF 的 chunk 数量")
print("=" * 50)
for fname, count in sorted(by_file.items(), key=lambda x: -x[1]):
    print(f"  {fname}: {count} 个 chunk")

# 重点看 lora.pdf
print("\n" + "=" * 50)
print("📊 lora.pdf 的 chunk 按页分布")
print("=" * 50)
lora_pages = Counter(
    m.get("page") for m in metadatas
    if m.get("source_file") == "lora.pdf"
)
for page in sorted(lora_pages.keys()):
    bar = "█" * lora_pages[page]
    print(f"  page {page:3d}: {lora_pages[page]:3d} {bar}")

# 看 lora.pdf 第 1 页附近的 chunk 内容
print("\n" + "=" * 50)
print("📋 lora.pdf 第 1 页的 chunks 内容预览")
print("=" * 50)
results = vs.get(
    where={"$and": [{"source_file": "lora.pdf"}, {"page": 1}]},
    include=["documents", "metadatas"],
)
for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
    print(f"\n--- chunk {meta.get('chunk_id')} ---")
    print(f"section: {meta.get('section')}")
    print(f"前 200 字符: {doc[:200]}")
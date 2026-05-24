"""
诊断: attention.pdf 第 3-4 页的 chunks 内容质量
看看为什么 HyDE 召回了第 5-6 页, 没召回讲核心公式的第 3-4 页
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import sys
sys.path.insert(0, "src")

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

# 拉取 attention.pdf 第 3-4 页的所有 chunks
print("=" * 70)
print("📋 attention.pdf 第 2、3、4 页的所有 chunks")
print("=" * 70)

for page in [2, 3, 4]:
    print(f"\n{'='*70}")
    print(f"📄 第 {page} 页")
    print(f"{'='*70}")
    
    results = vs.get(
        where={"$and": [{"source_file": "attention_is_all_you_need.pdf"}, {"page": page}]},
        include=["documents", "metadatas"],
    )
    
    if not results["documents"]:
        print(f"  (该页没有 chunks)")
        continue
    
    for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"]), 1):
        print(f"\n--- Chunk {i} (chunk_id={meta.get('chunk_id')}) ---")
        print(f"内容 ({len(doc)} 字符):")
        print(doc[:400])
        if len(doc) > 400:
            print(f"... [还有 {len(doc) - 400} 字符未显示]")
"""
RAG 主流程模块 v3
端到端 RAG 系统: 加载 → 切片 → 章节增强 → 向量化 → 混合检索 → Reranker → 生成

v3 升级 (vs v2):
- 集成混合检索 (BM25 30% + 向量 70%, RRF 融合)
- 集成 Reranker (bge-reranker-v2-m3 cross-encoder 精排)
- 元数据透传: source_file → page → section → chunk_id → retrieval_score → rerank_score
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from typing import List, Dict, Optional
from pathlib import Path

from langchain_core.documents import Document

from loader import load_pdfs
from section_detector import enrich_documents_with_sections
from splitter import split_documents
from retriever import build_vectorstore, search, build_hybrid_retriever, search_hybrid
from generator import generate_answer
from reranker import rerank_documents


class RAGPipeline:
    """
    端到端 RAG Pipeline v3 (混合检索 + Reranker)
    
    使用示例:
        rag = RAGPipeline()
        result = rag.query("LoRA 是什么?")
        print(result["answer"])
    """
    
    def __init__(
        self,
        pdf_dir: str = "data/raw_pdfs",
        persist_dir: str = "./chroma_db",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        candidate_k: int = 20,    # 召回阶段拉多少候选
        final_k: int = 5,          # 最终给 LLM 几个 chunk
        use_rerank: bool = True,   # 是否启用 Reranker
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        llm_model: str = "deepseek-chat",
        temperature: float = 0.3,
        force_rebuild: bool = False,
    ):
        self.pdf_dir = pdf_dir
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.candidate_k = candidate_k
        self.final_k = final_k
        self.use_rerank = use_rerank
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.llm_model = llm_model
        self.temperature = temperature
        
        print("🚀 初始化 RAG Pipeline v3 (混合检索 + Reranker)")
        print(f"   配置: candidate_k={candidate_k}, final_k={final_k}, use_rerank={use_rerank}")
        print(f"   权重: BM25={bm25_weight}, 向量={vector_weight}\n")
        
        # 准备 chunks 和向量库
        self.chunks, self.vectorstore = self._setup(force_rebuild)
        
        # 准备混合检索器
        self.hybrid_retriever = build_hybrid_retriever(
            self.chunks,
            self.vectorstore,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            candidate_k=candidate_k,
        )
        
        print("✅ RAG Pipeline 就绪\n")
    
    def _setup(self, force_rebuild: bool):
        """准备 chunks + 向量库"""
        # 加载 PDF (v2: PyPDF + pdfplumber)
        docs = load_pdfs(self.pdf_dir)
        
        # 注入章节元数据
        docs = enrich_documents_with_sections(docs)
        
        # 切片
        chunks = split_documents(
            docs,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        
        # 向量库
        vectorstore = build_vectorstore(
            chunks,
            persist_dir=self.persist_dir,
            force_rebuild=force_rebuild,
        )
        
        return chunks, vectorstore
    
    def retrieve(
        self,
        question: str,
        candidate_k: Optional[int] = None,
        final_k: Optional[int] = None,
        use_rerank: Optional[bool] = None,
    ) -> List[Document]:
        """
        混合检索 + (可选)Reranker 精排
        """
        ck = candidate_k or self.candidate_k
        fk = final_k or self.final_k
        do_rerank = self.use_rerank if use_rerank is None else use_rerank
        
        # Step 1: 混合检索召回
        candidates = search_hybrid(question, self.hybrid_retriever, top_k=ck)
        
        # Step 2: Reranker 精排(可选)
        if do_rerank:
            return rerank_documents(question, candidates, top_k=fk)
        else:
            return candidates[:fk]
    
    def query(
        self,
        question: str,
        candidate_k: Optional[int] = None,
        final_k: Optional[int] = None,
        use_rerank: Optional[bool] = None,
        return_context: bool = False,
    ) -> Dict:
        """
        端到端查询: 检索 + 精排 + 生成
        """
        # 检索 + 精排
        retrieved_docs = self.retrieve(
            question,
            candidate_k=candidate_k,
            final_k=final_k,
            use_rerank=use_rerank,
        )
        
        # 生成
        result = generate_answer(
            question,
            retrieved_docs,
            model=self.llm_model,
            temperature=self.temperature,
        )
        
        output = {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
        }
        if return_context:
            output["retrieved_docs"] = retrieved_docs
            output["context"] = result["context"]
        
        return output


# ==================== 测试入口 ====================
if __name__ == "__main__":
    # 一行初始化(默认开启 Reranker)
    rag = RAGPipeline()
    
    # 4 个核心测试问题(跟 Day 1/2/3 一致, 方便横向对比)
    test_questions = [
        "LoRA 是什么? 它如何减少可训练参数?",
        "什么是检索增强生成?它解决什么问题?",
        "Self-RAG 和普通 RAG 的区别是什么?",
        "Explain the self-attention mechanism in Transformer",
    ]
    
    for q in test_questions:
        print("=" * 70)
        print(f"❓ {q}")
        print("=" * 70)
        
        result = rag.query(q)
        
        print(f"\n💡 回答:\n{result['answer']}\n")
        print("📚 引用来源:")
        for s in result["sources"]:
            print(f"   - {s['file']} (第{s['page']}页)")
        print()
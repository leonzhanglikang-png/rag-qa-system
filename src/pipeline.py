"""
RAG 主流程模块
封装端到端 RAG 系统:加载 → 切片 → 向量化 → 检索 → 生成

外部只需要实例化 RAGPipeline 并调用 .query() 即可
"""
import os
from typing import List, Dict, Optional
from pathlib import Path

from langchain_core.documents import Document

from loader import load_pdfs
from splitter import split_documents
from retriever import build_vectorstore, search
from generator import generate_answer


class RAGPipeline:
    """
    端到端 RAG Pipeline
    
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
        top_k: int = 5,
        llm_model: str = "deepseek-chat",
        temperature: float = 0.3,
        force_rebuild: bool = False,
    ):
        """
        初始化 RAG Pipeline
        
        Args:
            pdf_dir: PDF 文档目录
            persist_dir: 向量库持久化目录
            chunk_size: 切片大小
            chunk_overlap: 切片重叠
            top_k: 检索返回的文档数
            llm_model: 使用的 LLM 模型
            temperature: 生成温度
            force_rebuild: 是否强制重建向量库
        """
        self.pdf_dir = pdf_dir
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.llm_model = llm_model
        self.temperature = temperature
        
        print("🚀 初始化 RAG Pipeline...")
        self.vectorstore = self._setup_vectorstore(force_rebuild)
        print("✅ RAG Pipeline 就绪\n")
    
    def _setup_vectorstore(self, force_rebuild: bool):
        """构建或加载向量库"""
        # 如果向量库不存在 或 强制重建,走完整 pipeline
        if force_rebuild or not Path(self.persist_dir).exists():
            print("📥 首次构建知识库,执行完整 pipeline...")
            docs = load_pdfs(self.pdf_dir)
            chunks = split_documents(
                docs,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            return build_vectorstore(
                chunks,
                persist_dir=self.persist_dir,
                force_rebuild=force_rebuild,
            )
        else:
            # 已存在,直接复用
            print("📂 加载已存在的向量库...")
            return build_vectorstore([], persist_dir=self.persist_dir)
    
    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[Document]:
        """
        只做检索,返回相关 chunks(不调 LLM, 节省成本)
        
        用于:① 调试检索效果  ② 评估时单独测检索召回率
        """
        k = top_k or self.top_k
        return search(question, self.vectorstore, top_k=k)
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_context: bool = False,
    ) -> Dict:
        """
        端到端查询:检索 + 生成
        
        Args:
            question: 用户问题
            top_k: 检索数量(不传用默认)
            return_context: 是否返回中间检索结果(调试用)
        
        Returns:
            {
                "question": 原问题,
                "answer": LLM 生成的答案,
                "sources": 引用的来源列表,
                "retrieved_docs": (可选) 检索到的原始 chunks,
            }
        """
        # 1. 检索
        retrieved_docs = self.retrieve(question, top_k=top_k)
        
        # 2. 生成
        result = generate_answer(
            question,
            retrieved_docs,
            model=self.llm_model,
            temperature=self.temperature,
        )
        
        # 3. 整理输出
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
    # 一行初始化
    rag = RAGPipeline()
    
    # 测试问题
    test_questions = [
        "LoRA 是什么? 它如何减少可训练参数?",
        "What is the core idea of Chain-of-Thought prompting?",
        "Self-RAG 和普通 RAG 的区别是什么?",
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
#ceshi
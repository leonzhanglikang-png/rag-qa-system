"""
RAG 评估模块: 用 Ragas 框架跑 4 个核心指标

指标:
- Context Precision: 召回的 chunk 中相关的比例(衡量检索准确度)
- Context Recall: 标准答案的关键事实在召回 chunk 中覆盖比例(衡量检索完整性)
- Faithfulness: LLM 答案能在 chunk 中找到根据的比例(反幻觉)
- Answer Relevancy: 答案是否真的回答了原问题(衡量是否跑题)

实现要点:
- Ragas 默认调 OpenAI, 我们替换为 DeepSeek (兼容 OpenAI API 格式)
- Ragas 默认用 OpenAI embedding, 替换为本地 bge-small-zh
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

import torch
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()


def get_judge_llm():
    """获取评估用的 LLM (DeepSeek, 兼容 OpenAI API)"""
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0.0,  # 评估要稳定, 用 0 温度
    )
    return LangchainLLMWrapper(llm)


def get_judge_embeddings():
    """获取评估用的 embedding (复用 bge-small-zh)"""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    return LangchainEmbeddingsWrapper(emb)


def collect_rag_outputs(rag_pipeline, questions: List[Dict]) -> List[Dict]:
    """
    用 RAG 系统跑所有评估问题, 收集输出
    
    Returns:
        每个问题的: question / answer / contexts / ground_truth(关键事实拼接)
    """
    outputs = []
    
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q['id']}: {q['question'][:50]}...")
        
        # 跑 RAG, return_context=True 拿到检索到的 chunks
        result = rag_pipeline.query(q["question"], return_context=True)
        
        # Ragas 需要的字段格式
        outputs.append({
            "question": q["question"],
            "answer": result["answer"],
            "contexts": [doc.page_content for doc in result["retrieved_docs"]],
            "ground_truth": ". ".join(q.get("key_facts", [])) or "N/A",
            "id": q["id"],
            "category": q["category"],
        })
    
    return outputs


def run_ragas_evaluation(outputs: List[Dict]) -> Dict:
    """
    跑 Ragas 评估, 输出 4 个指标
    """
    # 转成 Ragas 需要的 Dataset 格式
    dataset = Dataset.from_dict({
        "question": [o["question"] for o in outputs],
        "answer": [o["answer"] for o in outputs],
        "contexts": [o["contexts"] for o in outputs],
        "ground_truth": [o["ground_truth"] for o in outputs],
    })
    
    print(f"\n🤖 调用 Ragas 评估 ({len(dataset)} 个样本)...")
    print("   每个样本会调多次 LLM 做评判, 预计 5-15 分钟\n")
    
    # 跑评估
    judge_llm = get_judge_llm()
    judge_emb = get_judge_embeddings()
    
    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=judge_llm,
        embeddings=judge_emb,
    )
    
    return result


def save_eval_results(
    outputs: List[Dict],
    metrics: Dict,
    save_dir: str = "evaluation/results",
):
    """保存评估结果"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 保存原始 outputs
    with open(f"{save_dir}/outputs_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    
    # 保存指标
    metrics_dict = {k: float(v) for k, v in metrics._repr_dict.items()} if hasattr(metrics, '_repr_dict') else dict(metrics)
    with open(f"{save_dir}/metrics_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存:")
    print(f"   {save_dir}/outputs_{timestamp}.json")
    print(f"   {save_dir}/metrics_{timestamp}.json")
    
    return timestamp


# ==================== 测试入口 ====================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    
    from pipeline import RAGPipeline
    from eval_loader import load_eval_dataset
    
    # 加载评估集 (排除 D 类 - 它们单独评估"诚实性")
    print("=" * 70)
    print("📦 加载评估数据集")
    print("=" * 70)
    all_questions = load_eval_dataset()
    eval_questions = [q for q in all_questions if q["category"] != "out_of_scope"]
    print(f"  总问题: {len(all_questions)}")
    print(f"  评估集: {len(eval_questions)} (排除 out_of_scope 类)")
    
    # 初始化 RAG (开启全部优化: HyDE + 混合检索 + Reranker)
    print("\n" + "=" * 70)
    print("🚀 初始化 RAG Pipeline (全功能)")
    print("=" * 70)
    rag = RAGPipeline(use_hyde=True, use_rerank=True)
    
    # 跑 RAG 收集输出
    print("\n" + "=" * 70)
    print("🔄 跑 RAG 收集 15 个问题的回答 (预计 5-10 分钟)")
    print("=" * 70)
    outputs = collect_rag_outputs(rag, eval_questions)
    
    # 跑 Ragas 评估
    print("\n" + "=" * 70)
    print("📊 跑 Ragas 评估")
    print("=" * 70)
    metrics = run_ragas_evaluation(outputs)
    
    # 输出结果
    print("\n" + "=" * 70)
    print("🎯 评估结果")
    print("=" * 70)
    print(metrics)
    
    # 保存
    save_eval_results(outputs, metrics)
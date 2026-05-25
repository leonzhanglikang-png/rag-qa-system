"""
对比实验: v1 (baseline) vs v3 (no HyDE) vs v3+HyDE

跑同一个评估集 (15 个问题, 排除 D 类) 在三种配置下的 Ragas 指标
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, "src")

from pipeline import RAGPipeline
from eval_loader import load_eval_dataset
from evaluator import collect_rag_outputs, run_ragas_evaluation, save_eval_results


def run_one_config(config_name: str, rag_kwargs: dict, questions):
    """跑一个配置的完整评估"""
    print("\n" + "=" * 70)
    print(f"🔧 配置: {config_name}")
    print(f"   参数: {rag_kwargs}")
    print("=" * 70)
    
    # 初始化 RAG
    rag = RAGPipeline(**rag_kwargs)
    
    # 跑 RAG 收集回答
    print(f"\n🔄 收集 {len(questions)} 个问题的回答...")
    outputs = collect_rag_outputs(rag, questions)
    
    # 跑 Ragas 评估
    print(f"\n📊 跑 Ragas 评估...")
    metrics = run_ragas_evaluation(outputs)
    
    # 转字典(Ragas 返回的对象不能直接 json.dump)
    if hasattr(metrics, '_repr_dict'):
        metrics_dict = {k: float(v) for k, v in metrics._repr_dict.items()}
    else:
        metrics_dict = {k: float(v) for k, v in dict(metrics).items()}
    
    return outputs, metrics_dict


def main():
    # 加载评估集 (排除 D 类)
    all_questions = load_eval_dataset()
    eval_questions = [q for q in all_questions if q["category"] != "out_of_scope"]
    print(f"\n📦 评估集: {len(eval_questions)} 个问题")
    
    # 3 个配置
    configs = {
        "v1_baseline": {
            "use_hyde": False,
            "use_rerank": False,
        },
        "v3_no_hyde": {
            "use_hyde": False,
            "use_rerank": True,
        },
        "v3_full": {
            "use_hyde": True,
            "use_rerank": True,
        },
    }
    
    all_results = {}
    
    for name, kwargs in configs.items():
        outputs, metrics = run_one_config(name, kwargs, eval_questions)
        all_results[name] = {
            "config": kwargs,
            "metrics": metrics,
            "outputs": outputs,
        }
        
        print(f"\n🎯 {name} 结果:")
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}")
    
    # 保存完整对比结果
    save_dir = Path("evaluation/results")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    with open(save_dir / f"comparison_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # 打印对比表
    print("\n" + "=" * 70)
    print("📊 三个版本对比")
    print("=" * 70)
    
    metric_names = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    print(f"\n{'指标':<22} | {'v1 baseline':<12} | {'v3 no HyDE':<12} | {'v3 full':<12} | {'v1→v3':<10} | {'v3→full':<10}")
    print("-" * 110)
    
    for metric in metric_names:
        v1 = all_results["v1_baseline"]["metrics"].get(metric, 0)
        v3 = all_results["v3_no_hyde"]["metrics"].get(metric, 0)
        full = all_results["v3_full"]["metrics"].get(metric, 0)
        delta_1_3 = (v3 - v1) * 100
        delta_3_full = (full - v3) * 100
        print(f"{metric:<22} | {v1:.4f}      | {v3:.4f}      | {full:.4f}      | {delta_1_3:+.2f}pp    | {delta_3_full:+.2f}pp")
    
    print(f"\n💾 完整结果已保存: {save_dir}/comparison_{timestamp}.json")


if __name__ == "__main__":
    main()
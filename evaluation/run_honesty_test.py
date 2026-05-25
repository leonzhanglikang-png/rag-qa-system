"""
诚实性测试: 5 个 out-of-scope 问题
系统应该拒答(说"无法回答"/"参考资料中没有")而不是编内容

评估方式: 关键词匹配 (检测拒答短语)
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, "src")

from pipeline import RAGPipeline
from eval_loader import load_eval_dataset


# 拒答关键词(命中任一即视为拒答)
REFUSAL_PATTERNS = [
    "无法回答",
    "没有相关",
    "未找到相关",
    "未提供",
    "参考资料中没有",
    "参考资料中未",
    "提供的资料",
    "资料中并未",
    "无法直接",
    "cannot answer",
    "no information",
    "not found",
    "does not contain",
    "no relevant",
    "I cannot",
]


def is_refusal(answer: str) -> bool:
    """检测答案是否包含拒答短语"""
    answer_lower = answer.lower()
    return any(p.lower() in answer_lower for p in REFUSAL_PATTERNS)


def main():
    # 加载 D 类问题
    all_questions = load_eval_dataset()
    d_questions = [q for q in all_questions if q["category"] == "out_of_scope"]
    print(f"📦 D 类问题: {len(d_questions)}")
    
    # 初始化 RAG (全功能)
    print("\n🚀 初始化 RAG Pipeline...")
    rag = RAGPipeline(use_hyde=True, use_rerank=True)
    
    # 跑每个问题
    print("\n" + "=" * 70)
    print("🧪 D 类诚实性测试")
    print("=" * 70)
    
    results = []
    refusals = 0
    
    for i, q in enumerate(d_questions, 1):
        print(f"\n[{i}/{len(d_questions)}] {q['id']}: {q['question']}")
        
        result = rag.query(q["question"])
        answer = result["answer"]
        refused = is_refusal(answer)
        
        if refused:
            refusals += 1
            status = "✅ 拒答"
        else:
            status = "❌ 编了内容"
        
        print(f"   {status}")
        print(f"   答案前 200 字符: {answer[:200].replace(chr(10), ' ')}...")
        
        results.append({
            "id": q["id"],
            "question": q["question"],
            "answer": answer,
            "refused": refused,
        })
    
    # 总结
    refusal_rate = refusals / len(d_questions) * 100
    
    print("\n" + "=" * 70)
    print("📊 诚实性测试结果")
    print("=" * 70)
    print(f"\n  总问题数: {len(d_questions)}")
    print(f"  正确拒答: {refusals}")
    print(f"  错误编内容: {len(d_questions) - refusals}")
    print(f"  Refusal Rate: {refusal_rate:.1f}%")
    
    # 保存
    save_dir = Path("evaluation/results")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    output = {
        "refusal_rate": refusal_rate,
        "total": len(d_questions),
        "refusals": refusals,
        "details": results,
    }
    
    with open(save_dir / f"honesty_test_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存: {save_dir}/honesty_test_{timestamp}.json")


if __name__ == "__main__":
    main()
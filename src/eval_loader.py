"""
评估数据集加载工具
"""
import json
from pathlib import Path
from typing import List, Dict, Optional


def load_eval_dataset(
    path: str = "evaluation/eval_dataset.json",
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    加载评估数据集
    
    Args:
        path: JSON 文件路径
        category: 只加载指定类别 (single_doc_factual/cross_doc_comparison/chinese_query/out_of_scope)
        difficulty: 只加载指定难度 (easy/medium/hard)
        limit: 只加载前 N 个
    
    Returns:
        问题列表
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"评估数据集不存在: {path}")
    
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = data["questions"]
    
    if category:
        questions = [q for q in questions if q["category"] == category]
    if difficulty:
        questions = [q for q in questions if q["difficulty"] == difficulty]
    if limit:
        questions = questions[:limit]
    
    return questions


def dataset_stats(questions: List[Dict]) -> Dict:
    """统计数据集分布"""
    from collections import Counter
    
    return {
        "总数": len(questions),
        "类别分布": dict(Counter(q["category"] for q in questions)),
        "难度分布": dict(Counter(q["difficulty"] for q in questions)),
    }


if __name__ == "__main__":
    # 加载全部
    questions = load_eval_dataset()
    
    print("=" * 60)
    print("📊 评估数据集统计")
    print("=" * 60)
    stats = dataset_stats(questions)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 60)
    print("📋 前 5 个问题预览")
    print("=" * 60)
    for q in questions[:5]:
        print(f"\n[{q['id']}] ({q['category']}, {q['difficulty']})")
        print(f"  问题: {q['question']}")
        print(f"  期望来源: {q.get('expected_source')}")
        if q.get("key_facts"):
            print(f"  关键事实: {q['key_facts']}")
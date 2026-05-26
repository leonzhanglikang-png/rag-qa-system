"""
RAG QA System - Gradio Web Interface (Enhanced UI)

启动:
    python app.py
浏览器:
    http://localhost:7860
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import sys
sys.path.insert(0, "src")

import gradio as gr
from pipeline import RAGPipeline


# ==================== 初始化 ====================
print("=" * 70)
print("🚀 初始化 RAG Pipeline...")
print("=" * 70)
rag = RAGPipeline(use_hyde=True, use_rerank=True)
print("\n✅ RAG Pipeline 就绪, 启动 Gradio...")


# ==================== 业务逻辑 ====================
def query_rag(
    question: str,
    use_hyde: bool,
    use_rerank: bool,
    top_k: int,
):
    """流式 RAG 查询: yield 实时部分答案"""
    if not question.strip():
        yield "请输入问题", "", ""
        return
    
    # 初始状态: 显示"正在检索..."
    yield "🔍 正在检索相关资料...", "", ""
    
    # 流式生成
    last_result = None
    for partial in rag.query_streaming(
        question,
        use_hyde=use_hyde,
        use_rerank=use_rerank,
        final_k=int(top_k),
    ):
        last_result = partial
        
        # 流式更新: 当前累积的部分答案
        partial_answer = partial["answer"]
        
        # 来源 (生成完之前就先展示, 让用户知道在引用什么)
        sources_md = "**📚 引用来源**\n\n"
        for i, src in enumerate(partial["sources"], 1):
            sources_md += f"{i}. **{src['file']}** 第 {src['page']} 页\n"
        
        # 流式过程中暂不更新检索详情 (省 yield 性能)
        yield partial_answer, sources_md, ""
    
    # 全部完成后, 最后更新一次检索详情
    if last_result:
        retrieval_md = f"**🔍 检索详情** (Top-{int(top_k)} chunks, 已经 Reranker 精排)\n\n"
        for i, doc in enumerate(last_result["retrieved_docs"], 1):
            source = doc.metadata.get("source_file", "?")
            page = doc.metadata.get("page", "?")
            section = doc.metadata.get("section", "?") or "?"
            rerank_score = doc.metadata.get("rerank_score", "N/A")
            preview = doc.page_content[:200].replace("\n", " ")
            
            retrieval_md += f"### Chunk {i}: {source} 第 {page} 页\n"
            retrieval_md += f"**章节**: {section[:50]}  \n"
            retrieval_md += f"**Reranker 分数**: {rerank_score}  \n"
            retrieval_md += f"**内容预览**: {preview}...\n\n---\n\n"
        
        yield partial["answer"], sources_md, retrieval_md


# ==================== Gradio UI ====================
with gr.Blocks(
    title="RAG QA System",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown("# 📚 AI Papers RAG QA System")
    gr.Markdown(
        "知识库: 10 篇 AI 经典论文 / 294 页 / 2454 chunks / 126 表格  \n"
        "技术栈: BM25 + 向量检索 (混合) + Cross-encoder Reranker + HyDE 查询改写"
    )
    
    with gr.Row():
        # === 左侧 (主要交互区) ===
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="❓ 你的问题",
                placeholder="例如: LoRA 是什么? 它如何减少可训练参数?",
                lines=2,
            )
            submit_btn = gr.Button("🚀 提交", variant="primary", size="lg")
            
            answer_output = gr.Markdown(
                value="### 💡 回答\n*请提交问题...*",
                label="回答",
            )
            sources_output = gr.Markdown(value="")
            
            with gr.Accordion("🔍 查看检索详情 (点击展开)", open=False):
                retrieval_output = gr.Markdown(value="*提交问题后查看检索详情...*")
        
        # === 右侧 (配置 + 示例) ===
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 配置")
            use_hyde_checkbox = gr.Checkbox(
                label="启用 HyDE 查询改写",
                value=True,
                info="对中文/抽象查询有显著提升",
            )
            use_rerank_checkbox = gr.Checkbox(
                label="启用 Reranker 精排",
                value=True,
                info="Cross-encoder 精排, 大幅提升 Precision",
            )
            top_k_slider = gr.Slider(
                minimum=3,
                maximum=10,
                value=5,
                step=1,
                label="Top-K (送给 LLM 的 chunk 数)",
            )
            
            gr.Markdown("### 💡 示例问题")
            gr.Markdown("**单文档问答**")
            
            single_doc_examples = [
                "LoRA 是什么? 它如何减少可训练参数?",
                "QLoRA 相比 LoRA 主要改进了什么?",
                "Explain the self-attention mechanism in Transformer",
            ]
            for q in single_doc_examples:
                btn = gr.Button(q, size="sm")
                btn.click(lambda x=q: x, outputs=question_input)
            
            gr.Markdown("**跨文档对比**")
            cross_doc_examples = [
                "Self-RAG 和普通 RAG 的区别是什么?",
                "Compare BERT and the original Transformer",
            ]
            for q in cross_doc_examples:
                btn = gr.Button(q, size="sm")
                btn.click(lambda x=q: x, outputs=question_input)
            
            gr.Markdown("**中文跨语言**")
            chinese_examples = [
                "什么是检索增强生成?它解决什么问题?",
                "思维链 (Chain-of-Thought) 提示如何提升推理能力?",
            ]
            for q in chinese_examples:
                btn = gr.Button(q, size="sm")
                btn.click(lambda x=q: x, outputs=question_input)
    
    # === 绑定提交事件 ===
    submit_btn.click(
        fn=query_rag,
        inputs=[
            question_input,
            use_hyde_checkbox,
            use_rerank_checkbox,
            top_k_slider,
        ],
        outputs=[answer_output, sources_output, retrieval_output],
    )
    question_input.submit(
        fn=query_rag,
        inputs=[
            question_input,
            use_hyde_checkbox,
            use_rerank_checkbox,
            top_k_slider,
        ],
        outputs=[answer_output, sources_output, retrieval_output],
    )


# ==================== 启动 ====================
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )
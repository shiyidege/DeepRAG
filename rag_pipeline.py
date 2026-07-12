"""RAG 查询流水线

结合向量检索 + DeepSeek API 实现检索增强生成。
"""

import logging
from typing import List, Dict, Optional

from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    TOP_K,
)

logger = logging.getLogger(__name__)

# RAG 系统提示词
SYSTEM_PROMPT = """你是一个专业的知识问答助手。请基于提供的参考资料回答用户问题。

要求：
1. 只使用参考资料中的信息作答，不要编造事实
2. 如果参考资料不足以回答问题，请明确说明
3. 回答要简洁、准确、有条理
4. 适当引用参考来源

参考资料：
{context}"""


class RAGPipeline:
    """RAG 查询流水线"""

    def __init__(self, vector_store, api_key: str = None,
                 base_url: str = None, model: str = None):
        self.vector_store = vector_store
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.base_url = base_url or DEEPSEEK_BASE_URL
        self.model = model or DEEPSEEK_MODEL

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        logger.info(f"RAG Pipeline 初始化: model={self.model}")

    def query(self, question: str, k: int = None,
              with_context: bool = True) -> Dict:
        """执行 RAG 查询

        Args:
            question: 用户问题
            k: 检索的文档数
            with_context: 是否使用检索增强

        Returns:
            {"answer": str, "context": [...], "usage": {...}}
        """
        k = k or TOP_K

        # 1. 检索相关文档
        retrieved = self.vector_store.search(question, k=k)
        context_chunks = retrieved

        # 2. 构建 prompt
        if with_context and context_chunks:
            context_text = self._format_context(context_chunks)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(
                    context=context_text
                )},
                {"role": "user", "content": question},
            ]
        else:
            messages = [
                {"role": "system", "content": "你是一个专业的知识问答助手，请准确回答问题。"},
                {"role": "user", "content": question},
            ]

        # 3. 调用 DeepSeek
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )

        return {
            "answer": response.choices[0].message.content,
            "context": context_chunks,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

    def query_no_rag(self, question: str) -> Dict:
        """无 RAG 的纯 LLM 查询（用于对比实验）"""
        return self.query(question, with_context=False)

    def _format_context(self, chunks: List[Dict]) -> str:
        """格式化检索结果用于 prompt"""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            title = chunk["metadata"].get("title", "未知来源")
            parts.append(f"[{i}] 来源: {title}\n{chunk['text']}\n")
        return "\n---\n".join(parts)

    def stream_query(self, question: str, k: int = None):
        """流式 RAG 查询（用在 Colab 中实时显示）"""
        k = k or TOP_K
        retrieved = self.vector_store.search(question, k=k)

        context_text = self._format_context(retrieved)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                context=context_text
            )},
            {"role": "user", "content": question},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
            stream=True,
        )

        return response, context_text[:500]  # 返回流和截断的上下文预览

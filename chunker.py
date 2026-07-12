"""文本分块器

提供多级递归分块策略，针对中文文本优化：
1. 按标题/章节分割（最大块）
2. 按段落分割（\n\n）
3. 按句子分割（。！？）
4. 按 token 数截断（最小粒度）

支持自定义 max_tokens 和 overlap，用于对比实验。
"""

import re
import logging
from typing import List, Dict, Optional

import tiktoken

from config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNK_MIN_TOKENS

logger = logging.getLogger(__name__)


class ChineseTextChunker:
    """中文文本分块器，支持多级递归分割"""

    def __init__(self, max_tokens: int = CHUNK_MAX_TOKENS,
                 overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
                 min_tokens: int = CHUNK_MIN_TOKENS):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens
        # cl100k_base 是 GPT-4/DeepSeek 使用的 tokenizer
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数"""
        return len(self.encoder.encode(text))

    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """递归分块：标题 → 段落 → 句子 → 截断

        Args:
            text: 要分块的文本
            metadata: 附加元数据（标题、来源等）

        Returns:
            [{"text": str, "tokens": int, "metadata": Dict}, ...]
        """
        chunks = self._recursive_split(text)
        # 过滤过短的块，并添加元数据
        result = []
        for i, chunk_text in enumerate(chunks):
            token_count = self.count_tokens(chunk_text)
            if token_count < self.min_tokens:
                continue
            chunk_meta = dict(metadata or {})
            chunk_meta.update({
                "chunk_id": f"chunk_{i:04d}",
                "chunk_index": i,
            })
            result.append({
                "text": chunk_text,
                "tokens": token_count,
                "metadata": chunk_meta,
            })
        return result

    def _recursive_split(self, text: str) -> List[str]:
        """递归分割文本，逐步降低分割粒度"""
        # 如果整体不超过 max_tokens，直接返回
        if self.count_tokens(text) <= self.max_tokens:
            return [text]

        # 策略1: 按章节标题分割（## 或 === ... ===）
        sections = self._split_by_headings(text)
        if len(sections) > 1:
            return self._merge_small_chunks(sections)

        # 策略2: 按段落分割（连续换行）
        paragraphs = self._split_by_paragraphs(text)
        if len(paragraphs) > 1:
            return self._merge_small_chunks(paragraphs)

        # 策略3: 按句子分割（。！？）
        sentences = self._split_by_sentences(text)
        if len(sentences) > 1:
            return self._merge_small_chunks(sentences)

        # 策略4: 按 token 数截断 + 重叠
        return self._split_by_tokens(text)

    def _split_by_headings(self, text: str) -> List[str]:
        """按中文章节标题分割"""
        # 匹配 "## 标题" 或 "=== 标题 ===" 或 "第X章" 等
        patterns = [
            r'\n#{2,4}\s+[^\n]+',        # ## 或 ### 或 #### 标题
            r'\n===+[^=]+===+',           # === 标题 ===
            r'\n第[一二三四五六七八九十百千]+[章节篇部]',  # 第一章/节
        ]
        for pattern in patterns:
            parts = re.split(pattern, text)
            if len(parts) > 1:
                return [p.strip() for p in parts if p.strip()]
        return [text]

    def _split_by_paragraphs(self, text: str) -> List[str]:
        """按段落分割（连续换行分隔）"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_by_sentences(self, text: str) -> List[str]:
        """按中文句子分割"""
        # 中文句子结束符：。！？；
        sentences = re.split(r'(?<=[。！？；])\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """合并过小的块 + 切割过大的块"""
        merged = []
        buffer = ""

        for chunk in chunks:
            buffer_tokens = self.count_tokens(buffer)
            chunk_tokens = self.count_tokens(chunk)

            if buffer_tokens + chunk_tokens <= self.max_tokens:
                buffer = (buffer + "\n\n" + chunk).strip()
            else:
                if buffer:
                    merged.append(buffer)
                # 如果当前块仍然太大，递归分割
                if chunk_tokens > self.max_tokens:
                    sub_chunks = self._recursive_split(chunk)
                    merged.extend(sub_chunks)
                else:
                    buffer = chunk

        if buffer:
            merged.append(buffer)

        return merged

    def _split_by_tokens(self, text: str) -> List[str]:
        """按 token 数截断，带重叠"""
        tokens = self.encoder.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(chunk_text)
            start += self.max_tokens - self.overlap_tokens

        return chunks

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """批量分块文档列表

        Args:
            documents: [{"title": str, "content": str, "topics": [...], ...}, ...]

        Returns:
            [{"text": str, "tokens": int, "metadata": {...}}, ...]
        """
        all_chunks = []
        for doc in documents:
            metadata = {
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "source": doc.get("source", ""),
                "topics": ",".join(doc.get("topics", [])),
            }
            chunks = self.chunk(doc.get("content", ""), metadata)
            all_chunks.extend(chunks)
            logger.debug(f"  {doc.get('title', '')}: {len(chunks)} 块")

        logger.info(f"分块完成: {len(documents)} 篇 → {len(all_chunks)} 块")
        return all_chunks

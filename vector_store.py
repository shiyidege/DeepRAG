"""向量存储与检索

基于 ChromaDB + sentence-transformers 实现。
支持：
- 稠密检索（余弦相似度）
- 可选：稠密 + BM25 混合检索
- 可选：Cross-Encoder 重排序

支持检索策略对比实验。
"""

import logging
from typing import List, Dict, Optional, Callable

import numpy as np
import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_DIR, EMBEDDING_MODEL, TOP_K, RERANK_TOP_K

logger = logging.getLogger(__name__)


class VectorStore:
    """向量存储与检索"""

    def __init__(self, collection_name: str = "rag_docs",
                 persist_dir: str = None,
                 embedding_model: str = None):
        self.persist_dir = persist_dir or str(CHROMA_DIR)
        self.embedding_model_name = embedding_model or EMBEDDING_MODEL

        # 初始化 ChromaDB 客户端（自动持久化）
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # 初始化 embedding 函数
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )
        logger.info(f"Embedding 模型: {self.embedding_model_name}")

        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ef,
            metadata={"description": f"RAG with {embedding_model}"},
        )
        self._collection_name = collection_name

    @property
    def count(self) -> int:
        """当前存储的文档数"""
        return self.collection.count()

    def add_chunks(self, chunks: List[Dict]):
        """添加分块到向量数据库

        Args:
            chunks: [{"text": str, "tokens": int, "metadata": Dict}, ...]
        """
        if not chunks:
            logger.warning("没有要添加的块")
            return

        # 生成唯一 ID
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = chunk["metadata"].get("chunk_id", f"chunk_{i:06d}")
            # 添加全局唯一前缀
            ids.append(f"{self._collection_name}_{i:06d}_{chunk_id}")
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])

        # 批量添加（ChromaDB 自动计算 embedding）
        batch_size = 166  # ChromaDB 建议的批量大小
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )
            logger.debug(f"  已添加批次 {i}-{end}")

        logger.info(f"向量数据库: 添加 {len(chunks)} 个块 (总计 {self.count})")

    def search(self, query: str, k: int = None) -> List[Dict]:
        """标准稠密检索（余弦相似度）

        Args:
            query: 查询文本
            k: 返回 top-K 个结果

        Returns:
            [{"text": str, "metadata": Dict, "score": float}, ...]
        """
        k = k or TOP_K
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        return self._format_results(results)

    def search_with_rerank(self, query: str, k: int = None,
                           reranker_model: str = "BAAI/bge-reranker-v2-m3") -> List[Dict]:
        """稠密检索 + Cross-Encoder 重排序

        先检索 RERANK_TOP_K 个候选，再用 Cross-Encoder 重排序。
        """
        from sentence_transformers import CrossEncoder

        k = k or TOP_K
        # 第一阶段：取更多候选
        candidates = self.search(query, k=RERANK_TOP_K)
        if len(candidates) <= k:
            return candidates

        # 第二阶段：Cross-Encoder 重排序
        reranker = CrossEncoder(reranker_model)
        pairs = [(query, c["text"]) for c in candidates]
        scores = reranker.predict(pairs)

        # 按重排序分数降序排列
        ranked = list(zip(candidates, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)

        results = []
        for cand, score in ranked[:k]:
            cand["score"] = float(score)
            cand["rerank_score"] = float(score)
            results.append(cand)
        return results

    def hybrid_search(self, query: str, k: int = None,
                      dense_weight: float = 0.5) -> List[Dict]:
        """稠密 + 稀疏（BM25）混合检索

        融合余弦相似度和 BM25 分数，提升召回率。
        """
        from rank_bm25 import BM25Okapi

        k = k or TOP_K

        # 获取所有文档
        all_data = self.collection.get(include=["documents"])
        all_docs = all_data.get("documents", [])
        all_ids = all_data.get("ids", [])

        if not all_docs:
            return []

        # 稠密检索
        dense_results = self.search(query, k=len(all_docs))
        dense_scores = {r["id"]: r["score"] for r in dense_results}

        # BM25 检索
        tokenized_docs = [list(doc) for doc in all_docs]  # 按字分割
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = list(query)
        bm25_scores = bm25.get_scores(tokenized_query)

        # 归一化 BM25 分数
        bm25_max = max(bm25_scores) if bm25_scores.max() > 0 else 1
        bm25_norm = bm25_scores / bm25_max

        # 融合分数
        hybrid = []
        for i, doc_id in enumerate(all_ids):
            dense_score = dense_scores.get(doc_id, 0)
            bm25_score = float(bm25_norm[i])
            fused = dense_weight * dense_score + (1 - dense_weight) * bm25_score
            hybrid.append({
                "id": doc_id,
                "text": all_docs[i],
                "score": fused,
                "dense_score": dense_score,
                "bm25_score": bm25_score,
            })

        # 按融合分数排序
        hybrid.sort(key=lambda x: x["score"], reverse=True)
        return hybrid[:k]

    def _format_results(self, raw_results) -> List[Dict]:
        """格式化 ChromaDB 返回结果"""
        results = []
        if not raw_results or not raw_results.get("ids"):
            return results

        for i in range(len(raw_results["ids"][0])):
            result = {
                "id": raw_results["ids"][0][i],
                "text": raw_results["documents"][0][i],
                "metadata": raw_results["metadatas"][0][i],
                "score": 1 - raw_results["distances"][0][i] if raw_results.get("distances") else 0,
            }
            results.append(result)
        return results

    def get_stats(self) -> Dict:
        """数据库统计信息"""
        return {
            "total_chunks": self.count,
            "embedding_model": self.embedding_model_name,
            "persist_dir": self.persist_dir,
        }

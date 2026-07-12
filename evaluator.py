"""RAG 评估模块

双维度评估体系：
1. 检索评估：Recall@K, MRR, NDCG@K — 衡量检索质量
2. 生成评估：LLM-as-Judge — 衡量生成质量（相关性、忠实度、完整性）

评估流程：取文档分块 → DeepSeek 自动生成 QA 对 → 评估检索和生成。
"""

import json
import logging
import time
from typing import List, Dict, Callable, Optional

import numpy as np
from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    EVAL_N_QUESTIONS, EVAL_RETRIEVAL_K, OUTPUT_DIR,
)

logger = logging.getLogger(__name__)


class QAGenerator:
    """QA 对生成器

    从文档块中提取关键内容，使用 LLM 生成 (问题, 答案, 源块) 三元组。
    """

    def __init__(self):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )

    def generate(self, chunks: List[Dict], n: int = EVAL_N_QUESTIONS) -> List[Dict]:
        """生成 QA 对

        策略：随机采样 n 个块，对每个块用 LLM 生成一个问题。
        问题基于块内容生成，答案也在块中。

        Args:
            chunks: 文档块列表
            n: 生成的 QA 对数量

        Returns:
            [{"question": str, "answer": str, "source_chunk": str, "source_title": str}, ...]
        """
        if len(chunks) < n:
            n = len(chunks)
            logger.warning(f"块数不足，实际生成 {n} 个 QA 对")

        # 均匀采样：按块大小加权，越大的块越可能被选
        weights = [max(c["tokens"], 50) for c in chunks]
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()

        selected_indices = np.random.choice(
            len(chunks), size=n, replace=False, p=weights
        )

        qa_pairs = []
        for idx in selected_indices:
            chunk = chunks[idx]
            qa = self._generate_single(chunk)
            if qa:
                qa_pairs.append(qa)
            time.sleep(0.5)  # API 限流

        logger.info(f"生成 QA 对: {len(qa_pairs)}/{n}")
        return qa_pairs

    def _generate_single(self, chunk: Dict) -> Optional[Dict]:
        """为一个块生成一个 QA 对"""
        text = chunk["text"]
        title = chunk["metadata"].get("title", "未知")

        prompt = f"""基于以下文本，生成一个中文问答对。问题要有具体答案（不要问"是什么"这类泛泛的问题），答案要在文本中能找到明确的依据。

要求：
- 问题要具体，需要从文本中推理或查找才能回答
- 答案要准确，只使用文本中的信息
- 答案控制在 50-150 字

文本（来源：{title}）：
{text[:1500]}  # 截断避免超长

请按以下 JSON 格式输出（不要加其他内容）：
{{"question": "问题", "answer": "答案"}}"""

        try:
            resp = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个专业的 QA 生成器，输出严格的 JSON 格式。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            qa = json.loads(content)

            return {
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "source_chunk": text,
                "source_title": title,
                "source_tokens": chunk["tokens"],
            }
        except Exception as e:
            logger.warning(f"生成 QA 失败 ({title}): {e}")
            return None


class RetrievalEvaluator:
    """检索评估器

    指标：
    - Recall@K: 正确答案在前 K 个结果中的比例
    - MRR: 第一个正确答案排名的倒数均值
    - NDCG@K: 归一化折损累计增益
    """

    def __init__(self, vector_store):
        self.vs = vector_store

    def evaluate(self, qa_pairs: List[Dict], k_values: List[int] = None) -> Dict:
        """评估检索质量"""
        k_values = k_values or EVAL_RETRIEVAL_K
        max_k = max(k_values)

        all_recalls = {k: [] for k in k_values}
        all_mrrs = []
        all_ndcgs = {k: [] for k in k_values}

        for qa in qa_pairs:
            question = qa["question"]
            source_text = qa["source_chunk"]

            results = self.vs.search(question, k=max_k)
            retrieved_texts = [r["text"] for r in results]

            # 判断是否命中源块
            hit_positions = []
            for pos, text in enumerate(retrieved_texts):
                if self._is_match(source_text, text):
                    hit_positions.append(pos)

            # Recall@K
            for k in k_values:
                hit = any(p < k for p in hit_positions)
                all_recalls[k].append(1.0 if hit else 0.0)

            # MRR
            if hit_positions:
                all_mrrs.append(1.0 / (hit_positions[0] + 1))
            else:
                all_mrrs.append(0.0)

            # NDCG@K
            for k in k_values:
                ndcg = self._compute_ndcg(hit_positions, k)
                all_ndcgs[k].append(ndcg)

        results = {}
        for k in k_values:
            results[f"recall@{k}"] = {
                "value": float(np.mean(all_recalls[k])),
                "std": float(np.std(all_recalls[k])),
            }
            results[f"ndcg@{k}"] = {
                "value": float(np.mean(all_ndcgs[k])),
                "std": float(np.std(all_ndcgs[k])),
            }
        results["mrr"] = {
            "value": float(np.mean(all_mrrs)),
            "std": float(np.std(all_mrrs)),
        }

        logger.info(f"检索评估: Recall@5={results['recall@5']['value']:.3f}")
        return results

    def _is_match(self, source: str, retrieved: str) -> bool:
        """判断检索结果是否与源块匹配"""
        # 使用长公共子串判断（比完全匹配更鲁棒）
        source_stripped = source.strip()[:200]
        retrieved_stripped = retrieved.strip()[:200]
        return source_stripped in retrieved_stripped or retrieved_stripped in source_stripped

    def _compute_ndcg(self, hit_positions: List[int], k: int) -> float:
        """计算 NDCG@K"""
        relevance = [0] * k
        for pos in hit_positions:
            if pos < k:
                relevance[pos] = 1

        dcg = relevance[0] if relevance else 0
        for i in range(1, k):
            dcg += relevance[i] / np.log2(i + 1)

        # 理想 DCG（假设第一个就命中）
        idcg = 1.0
        return dcg / idcg if idcg > 0 else 0.0


class GenerationEvaluator:
    """生成评估器（LLM-as-Judge）

    指标：
    - 忠实度 Faithfulness (1-5): 答案是否基于提供的上下文，没有幻觉
    - 相关性 Relevance (1-5): 答案是否直接回应了问题
    - 完整性 Completeness (1-5): 答案是否包含了必要的信息
    """

    def __init__(self, rag_pipeline):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        self.rag = rag_pipeline

    def evaluate(self, qa_pairs: List[Dict], sample_size: int = None) -> Dict:
        """评估生成质量

        对每个 QA 对执行 RAG 查询，然后用 LLM 评分。
        """
        if sample_size:
            qa_pairs = qa_pairs[:sample_size]

        scores = {"faithfulness": [], "relevance": [], "completeness": []}
        details = []

        for i, qa in enumerate(qa_pairs):
            question = qa["question"]
            expected_answer = qa["answer"]

            # RAG 生成
            result = self.rag.query(question)
            generated_answer = result["answer"]
            context = result["context"]

            # LLM 评分
            score = self._judge(question, generated_answer, expected_answer, context)
            if score:
                for k in scores:
                    scores[k].append(score.get(k, 0))
                details.append({
                    "question": question,
                    "generated": generated_answer,
                    "expected": expected_answer,
                    **score,
                })

            if (i + 1) % 5 == 0:
                logger.info(f"  生成评估: {i + 1}/{len(qa_pairs)}")

            time.sleep(0.5)  # 限流

        results = {}
        for k, v in scores.items():
            if v:
                results[k] = {
                    "value": float(np.mean(v)),
                    "std": float(np.std(v)),
                }
            else:
                results[k] = {"value": 0, "std": 0}

        results["avg_score"] = {
            "value": float(np.mean([scores[k] for k in scores if scores[k]])),
            "std": 0,
        }
        results["details"] = details[:5]  # 只保留前 5 个详细记录

        logger.info(
            f"生成评估: 忠实度={results['faithfulness']['value']:.2f}, "
            f"相关性={results['relevance']['value']:.2f}"
        )
        return results

    def _judge(self, question: str, answer: str,
               expected: str, context: List[Dict]) -> Optional[Dict]:
        """LLM-as-Judge 评分"""
        context_text = "\n".join(
            [f"[{i+1}] {c['text'][:300]}" for i, c in enumerate(context)]
        ) if context else "无上下文"

        prompt = f"""你是一个 RAG 系统评估专家。请评估以下 RAG 生成的回答质量。

用户问题：{question}

参考上下文（检索到的文档）：
{context_text}

RAG 生成的回答：{answer}

参考标准答案（基于原文）：{expected}

请从三个维度评分（1-5分）：
1. faithful（忠实度）：回答是否严格基于提供的上下文，没有编造事实？
2. relevant（相关性）：回答是否直接回应了问题？
3. complete（完整性）：回答是否涵盖了问题的关键方面？

请以 JSON 格式输出，不要加其他内容：
{{"faithful": 5, "relevant": 4, "complete": 3}}"""

        try:
            resp = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个严格的 RAG 评估专家，输出 JSON 格式。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            scores = json.loads(content)
            return {
                "faithfulness": float(scores.get("faithful", 3)),
                "relevance": float(scores.get("relevant", 3)),
                "completeness": float(scores.get("complete", 3)),
            }
        except Exception as e:
            logger.warning(f"评分失败: {e}")
            return {"faithfulness": 0, "relevance": 0, "completeness": 0}


def plot_eval_results(retrieval_results: Dict, gen_results: Dict,
                      save_path: str = None) -> str:
    """可视化评估结果"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. 检索指标柱状图
    ax = axes[0]
    metrics = [k for k in retrieval_results.keys() if k != "type"]
    values = [retrieval_results[k]["value"] for k in metrics]
    stds = [retrieval_results[k].get("std", 0) for k in metrics]
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(metrics)))
    bars = ax.bar(metrics, values, yerr=stds, color=colors, capsize=5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Metrics", fontsize=12)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3)

    # 2. 生成指标雷达图
    ax = axes[1]
    if gen_results:
        categories = ["faithfulness", "relevance", "completeness"]
        values = [gen_results.get(c, {}).get("value", 0) for c in categories]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        ax.plot(angles, values, "o-", linewidth=2, color="green")
        ax.fill(angles, values, alpha=0.25, color="green")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 5)
        ax.set_title("Generation Quality (LLM-as-Judge)", fontsize=12)
        for angle, val in zip(angles[:-1], values[:-1]):
            ax.text(angle, val + 0.3, f"{val:.1f}", ha="center", fontsize=9)

    # 3. 综合分数
    ax = axes[2]
    ax.axis("off")
    info_text = "RAG Evaluation Summary\n\n"
    for k, v in retrieval_results.items():
        info_text += f"  {k}: {v['value']:.4f} ± {v['std']:.4f}\n"
    info_text += "\n"
    if gen_results:
        for k in ["faithfulness", "relevance", "completeness"]:
            v = gen_results.get(k, {})
            info_text += f"  {k}: {v.get('value', 0):.2f} ± {v.get('std', 0):.2f}\n"
    ax.text(0.1, 0.5, info_text, fontsize=11, verticalalignment="center",
            fontfamily="monospace", linespacing=1.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"评估图表已保存: {save_path}")
    plt.show()
    return save_path

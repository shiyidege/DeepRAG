"""RAG 项目主入口

使用方式:
    python main.py crawl          # 爬取维基百科数据
    python main.py process        # 分块 + 建索引
    python main.py query "问题"    # RAG 查询
    python main.py eval            # 评估检索 + 生成质量
    python main.py pipeline        # 完整流水线: crawl → process → eval
    python main.py stats           # 查看数据统计
"""

import sys
import json
import logging
import argparse
from pathlib import Path

from config import DATA_DIR, OUTPUT_DIR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_crawl(args):
    """爬取中文维基百科 AI 领域文章（失败时自动回退到内置知识库）"""
    from data_crawler import crawl_all_topics, save_articles, get_article_stats
    from fallback_knowledge import get_fallback_articles

    logger.info("=" * 50)
    logger.info("开始爬取维基百科数据")
    logger.info("=" * 50)

    articles = crawl_all_topics(max_articles=args.max_articles)

    # 如果维基百科爬取失败（被限流），自动使用内置知识库
    if len(articles) == 0:
        logger.warning("\n维基百科 API 不可用（可能被限流），自动切换到内置知识库")
        logger.info("使用内置 AI 知识库（13 篇精选文章）...")
        articles = get_fallback_articles()
        logger.info(f"内置知识库包含 {len(articles)} 篇文章")

    path = save_articles(articles)
    stats = get_article_stats(articles)
    logger.info(f"爬取完成:")
    logger.info(f"  文章数: {stats['count']}")
    logger.info(f"  总字符: {stats['total_chars']:,}")
    logger.info(f"  平均字符: {stats['avg_chars']}")


def cmd_process(args):
    """分块并构建向量索引"""
    from data_crawler import load_articles
    from chunker import ChineseTextChunker
    from vector_store import VectorStore

    logger.info("=" * 50)
    logger.info("处理数据：分块 + 建索引")
    logger.info("=" * 50)

    # 加载文章
    articles = load_articles()
    if not articles:
        logger.error("没有找到文章数据，请先运行 python main.py crawl")
        return

    logger.info(f"加载 {len(articles)} 篇文章")

    # 分块
    chunker = ChineseTextChunker(
        max_tokens=args.max_tokens,
        overlap_tokens=getattr(args, "overlap_tokens", 64),
    )
    chunks = chunker.chunk_documents(articles)
    logger.info(f"分块完成: {len(chunks)} 块")

    # 保存分块信息
    chunk_info = {
        "total_chunks": len(chunks),
        "max_tokens": args.max_tokens,
        "overlap_tokens": args.overlap_tokens,
        "avg_tokens": sum(c["tokens"] for c in chunks) / len(chunks),
        "total_tokens": sum(c["tokens"] for c in chunks),
    }
    with open(OUTPUT_DIR / "chunk_stats.json", "w", encoding="utf-8") as f:
        json.dump(chunk_info, f, ensure_ascii=False, indent=2)

    # 建索引
    vs = VectorStore(embedding_model=args.embedding)
    vs.add_chunks(chunks)

    stats = vs.get_stats()
    logger.info(f"索引完成: {stats['total_chunks']} 个块")
    logger.info(f"  分块参数: max_tokens={args.max_tokens}, overlap={args.overlap_tokens}")


def cmd_query(args):
    """执行 RAG 查询"""
    from vector_store import VectorStore
    from rag_pipeline import RAGPipeline

    logger.info("=" * 50)
    logger.info("RAG 查询")
    logger.info("=" * 50)

    # 加载向量库
    vs = VectorStore()
    if vs.count == 0:
        logger.error("向量库为空，请先运行 python main.py process")
        return

    # 初始化 RAG
    rag = RAGPipeline(vs)

    # 执行查询
    logger.info(f"问题: {args.question}")
    result = rag.query(args.question, k=args.top_k)

    # 输出结果
    print("\n" + "=" * 60)
    print("回答:")
    print(result["answer"])
    print("=" * 60)
    print(f"\n参考来源 ({len(result['context'])} 个):")
    for i, ctx in enumerate(result["context"], 1):
        title = ctx["metadata"].get("title", "未知")
        score = ctx["score"]
        print(f"  [{i}] {title} (相关度: {score:.3f})")
    print(f"\nToken 使用: {result['usage']}")


def cmd_eval(args):
    """评估 RAG 系统"""
    from data_crawler import load_articles
    from chunker import ChineseTextChunker
    from vector_store import VectorStore
    from rag_pipeline import RAGPipeline
    from evaluator import QAGenerator, RetrievalEvaluator, GenerationEvaluator, plot_eval_results

    logger.info("=" * 50)
    logger.info("RAG 系统评估")
    logger.info("=" * 50)

    # 加载文章
    articles = load_articles()
    if not articles:
        logger.error("没有找到文章数据，请先运行 python main.py crawl")
        return

    # 分块
    chunker = ChineseTextChunker(max_tokens=args.max_tokens)
    chunks = chunker.chunk_documents(articles)

    # 建索引
    vs = VectorStore(embedding_model=args.embedding)
    vs.add_chunks(chunks)
    logger.info(f"向量库: {vs.count} 个块")

    # 生成 QA 对
    logger.info("\n[1/3] 生成 QA 测试集...")
    qa_gen = QAGenerator()
    qa_pairs = qa_gen.generate(chunks, n=args.n_questions)
    logger.info(f"生成 {len(qa_pairs)} 个 QA 对")

    if not qa_pairs:
        logger.error("QA 对生成失败")
        return

    # 保存 QA 对
    qa_path = OUTPUT_DIR / "qa_pairs.json"
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    # 检索评估
    logger.info("\n[2/3] 评估检索质量...")
    ret_eval = RetrievalEvaluator(vs)
    ret_results = ret_eval.evaluate(qa_pairs)

    print("\n--- 检索指标 ---")
    for k, v in ret_results.items():
        print(f"  {k}: {v['value']:.4f} (std: {v['std']:.4f})")

    # 生成评估
    logger.info("\n[3/3] 评估生成质量 (LLM-as-Judge)...")
    rag = RAGPipeline(vs)
    gen_eval = GenerationEvaluator(rag)
    gen_results = gen_eval.evaluate(qa_pairs, sample_size=args.n_eval)

    print("\n--- 生成质量 ---")
    for k in ["faithfulness", "relevance", "completeness"]:
        v = gen_results.get(k, {})
        print(f"  {k}: {v.get('value', 0):.2f} (std: {v.get('std', 0):.2f})")

    # 可视化
    save_path = str(OUTPUT_DIR / "eval_results.png")
    plot_eval_results(ret_results, gen_results, save_path)

    # 保存评估结果
    eval_path = OUTPUT_DIR / "eval_results.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump({
            "retrieval": {k: {"value": round(v["value"], 4), "std": round(v["std"], 4)}
                          for k, v in ret_results.items()},
            "generation": {k: {"value": round(v["value"], 4), "std": round(v["std"], 4)}
                           for k, v in gen_results.items() if k not in ["details"]},
            "config": {"max_tokens": args.max_tokens, "n_questions": args.n_questions},
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"评估完成! 结果已保存到 {OUTPUT_DIR}")


def cmd_pipeline(args):
    """完整流水线: crawl → process → eval"""
    logger.info("=" * 60)
    logger.info("完整 RAG 流水线")
    logger.info("=" * 60)

    cmd_crawl(args)
    cmd_process(args)
    cmd_eval(args)


def cmd_stats(args):
    """查看数据统计"""
    from data_crawler import load_articles, get_article_stats

    articles = load_articles()
    stats = get_article_stats(articles)

    print("\n数据统计:")
    print(f"  文章数: {stats['count']}")
    print(f"  总字符数: {stats['total_chars']:,}")
    print(f"  平均每篇字符: {stats['avg_chars']:,}")
    print(f"  最长: {stats['max_chars']:,} 字符")
    print(f"  最短: {stats['min_chars']:,} 字符")

    # 检查向量库
    from vector_store import VectorStore
    vs = VectorStore()
    print(f"\n向量库:")
    print(f"  文档块数: {vs.count}")
    print(f"  Embedding 模型: {vs.embedding_model_name}")
    print(f"  持久化路径: {vs.persist_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="RAG 项目 - 检索增强生成系统"
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # crawl
    p = subparsers.add_parser("crawl", help="爬取维基百科 AI 领域文章")
    p.add_argument("--max_articles", type=int, default=50,
                   help="最大文章数")

    # process
    p = subparsers.add_parser("process", help="分块 + 建索引")
    p.add_argument("--max_tokens", type=int, default=512,
                   help="每块最大 token 数")
    p.add_argument("--overlap_tokens", type=int, default=64,
                   help="块间重叠 token 数")
    p.add_argument("--embedding", type=str, default="BAAI/bge-small-zh-v1.5",
                   help="Embedding 模型")

    # query
    p = subparsers.add_parser("query", help="RAG 查询")
    p.add_argument("question", type=str, help="问题")
    p.add_argument("--top_k", type=int, default=5,
                   help="检索 Top-K 个文档")

    # eval
    p = subparsers.add_parser("eval", help="评估 RAG 系统")
    p.add_argument("--max_tokens", type=int, default=512,
                   help="分块大小")
    p.add_argument("--embedding", type=str, default="BAAI/bge-small-zh-v1.5",
                   help="Embedding 模型")
    p.add_argument("--n_questions", type=int, default=30,
                   help="QA 测试集大小")
    p.add_argument("--n_eval", type=int, default=15,
                   help="生成评估样本数")

    # pipeline
    p = subparsers.add_parser("pipeline", help="完整流水线")
    p.add_argument("--max_articles", type=int, default=50)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--embedding", type=str, default="BAAI/bge-small-zh-v1.5")
    p.add_argument("--n_questions", type=int, default=30)
    p.add_argument("--n_eval", type=int, default=15)

    # stats
    subparsers.add_parser("stats", help="查看数据统计")

    args = parser.parse_args()

    if args.command == "crawl":
        cmd_crawl(args)
    elif args.command == "process":
        cmd_process(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "pipeline":
        cmd_pipeline(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""维基百科数据爬虫

从中文维基百科抓取指定主题的文章，保存为结构化 JSON 文件。
如果维基 API 不可用，提供退回到本地知识库的机制。
"""

import json
import time
import random
import logging
import requests
from pathlib import Path
from typing import List, Dict, Optional

from config import DATA_DIR, WIKI_TOPICS, MAX_ARTICLES, WIKI_USER_AGENT

logger = logging.getLogger(__name__)

WIKI_API = "https://zh.wikipedia.org/w/api.php"

# 随机 User-Agent 轮转，降低被限流的概率
USER_AGENTS = [
    WIKI_USER_AGENT,
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# 指数退避重试参数
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # 初始等待秒数


def _get_headers() -> dict:
    return {"User-Agent": random.choice(USER_AGENTS)}


def _request_with_retry(params: dict, max_retries: int = MAX_RETRIES) -> Optional[dict]:
    """带指数退避重试的 API 请求"""
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                WIKI_API, params=params,
                headers=_get_headers(),
                timeout=30,
            )
            if resp.status_code == 429:
                wait = INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"429 限流，等待 {wait:.1f}s 后重试 ({attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as e:
            wait = INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 2)
            logger.warning(f"连接错误: {e}，等待 {wait:.1f}s 后重试")
            time.sleep(wait)
        except requests.exceptions.Timeout:
            wait = INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 2)
            logger.warning(f"请求超时，等待 {wait:.1f}s 后重试")
            time.sleep(wait)
        except Exception as e:
            logger.warning(f"请求失败: {e}")
            return None
    logger.error(f"重试 {max_retries} 次后仍然失败")
    return None


def search_articles(topic: str, limit: int = 10) -> List[str]:
    """搜索中文维基百科文章标题"""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "srlimit": limit,
        "format": "json",
        "srprop": "",
    }
    data = _request_with_retry(params)
    if data:
        titles = [r["title"] for r in data.get("query", {}).get("search", [])]
        logger.debug(f"搜索 '{topic}' 找到 {len(titles)} 篇")
        return titles
    return []


def get_article_content(title: str) -> Optional[str]:
    """获取文章完整内容（纯文本）"""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
        "formatversion": 2,
        "redirects": 1,
    }
    data = _request_with_retry(params)
    if data:
        pages = data.get("query", {}).get("pages", [])
        if pages and pages[0].get("extract"):
            return pages[0]["extract"].strip()
    return None


def crawl_all_topics(topics: List[str] = None,
                     max_articles: int = None) -> List[Dict]:
    """爬取所有主题的文章

    Returns:
        [{title, url, content, topics, source}, ...]
    """
    topics = topics or WIKI_TOPICS
    max_articles = max_articles or MAX_ARTICLES

    all_titles = set()
    articles = []

    # 第一阶段：收集文章标题
    logger.info("第一阶段：搜索文章标题...")
    success_count = 0
    for topic in topics:
        titles = search_articles(topic)
        if titles:
            success_count += 1
        for t in titles:
            all_titles.add(t)
        time.sleep(random.uniform(0.8, 1.5))

    logger.info(f"共找到 {len(all_titles)} 篇不重复文章")
    if success_count == 0:
        logger.warning("所有搜索请求均失败（疑似被维基百科限流），跳过爬取")
        return []

    # 第二阶段：获取内容
    logger.info("第二阶段：获取文章内容...")
    title_list = list(all_titles)[:max_articles]

    for i, title in enumerate(title_list):
        logger.info(f"  [{i+1}/{len(title_list)}] {title}")
        content = get_article_content(title)
        if content and len(content) > 100:
            matched_topics = [t for t in topics if t in title or t in content[:200]]
            articles.append({
                "title": title,
                "url": f"https://zh.wikipedia.org/wiki/{requests.utils.quote(title)}",
                "content": content,
                "topics": matched_topics or ["其他"],
                "source": "zh.wikipedia.org",
                "char_count": len(content),
            })
            logger.info(f"    ✓ {len(content)} 字符")
        else:
            logger.info(f"    ✗ 内容为空或过短")
        time.sleep(random.uniform(0.8, 1.5))

    logger.info(f"成功爬取 {len(articles)} 篇文章")
    return articles


def save_articles(articles: List[Dict], path: Path = None) -> Path:
    """保存文章到 JSON 文件"""
    path = path or (DATA_DIR / "articles.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存 {len(articles)} 篇文章到 {path}")
    return path


def load_articles(path: Path = None) -> List[Dict]:
    """从 JSON 加载文章"""
    path = path or (DATA_DIR / "articles.json")
    if not path.exists():
        logger.warning(f"文件不存在: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    logger.info(f"已加载 {len(articles)} 篇文章")
    return articles


def get_article_stats(articles: List[Dict]) -> Dict:
    """文章统计信息"""
    if not articles:
        return {"count": 0, "total_chars": 0, "avg_chars": 0}
    total = sum(a["char_count"] for a in articles)
    return {
        "count": len(articles),
        "total_chars": total,
        "avg_chars": total // len(articles),
        "max_chars": max(a["char_count"] for a in articles),
        "min_chars": min(a["char_count"] for a in articles),
    }

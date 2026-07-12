"""全局配置"""

import os
from pathlib import Path

# ========== 路径 ==========
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"           # 爬取的原始文章
CHROMA_DIR = BASE_DIR / "chroma_db"    # 向量数据库持久化目录
OUTPUT_DIR = BASE_DIR / "output"       # 评估结果输出
LOG_DIR = BASE_DIR / "logs"            # 日志

for d in [DATA_DIR, CHROMA_DIR, OUTPUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ========== DeepSeek API ==========
DEEPSEEK_API_KEY = "sk-XXXXXX"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
# 长上下文模型用于 QA 生成等需要长文本的场景
DEEPSEEK_LONG_MODEL = "deepseek-chat"

# ========== Embedding 模型 ==========
# bge-small-zh-v1.5: 384维, 速度快, 中文效果好
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
# 备选: "shibing624/text2vec-base-chinese" (768维, 更准但更慢)
EMBEDDING_DIM = 384

# ========== 分块参数 ==========
CHUNK_MAX_TOKENS = 512          # 每块最大 token 数
CHUNK_OVERLAP_TOKENS = 64       # 块间重叠 token 数
CHUNK_MIN_TOKENS = 50           # 最小块大小（小于此丢弃）

# ========== 检索参数 ==========
TOP_K = 5                        # 检索返回 Top-K 个块
RERANK_TOP_K = 20                # 重排序前先取 Top-20

# ========== 维基百科爬取 ==========
WIKI_LANG = "zh"
WIKI_USER_AGENT = "RAG-Project/1.0 (Educational Research)"
# 要爬取的 AI 领域主题
WIKI_TOPICS = [
    "人工智能", "机器学习", "深度学习", "神经网络",
    "自然语言处理", "计算机视觉", "强化学习",
    "大语言模型", "卷积神经网络", "循环神经网络",
    "Transformer", "BERT", "GPT", "生成对抗网络",
    "迁移学习", "监督学习", "无监督学习",
    "支持向量机", "决策树", "随机森林", "聚类分析",
    "知识图谱", "推荐系统", "语音识别",
    "图像识别", "目标检测", "语义分割",
    "注意力机制", "自监督学习", "联邦学习",
    "多模态学习", "图神经网络", "扩散模型",
]
MAX_ARTICLES = 50                # 最多爬取文章数

# ========== 评估参数 ==========
EVAL_N_QUESTIONS = 30            # 生成多少测试 QA 对
EVAL_RETRIEVAL_K = [1, 3, 5]    # 评估 Recall@K 的 K 值

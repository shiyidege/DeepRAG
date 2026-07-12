# RAG-Project — 检索增强生成系统

基于 DeepSeek + ChromaDB + BGE Embedding 的 RAG 实现。从中文维基百科或自定义数据源构建 AI 知识库，支持高质量问答。

## 项目特点

- **完整流水线**：数据准备 → 多级分块 → 向量建索引 → RAG 查询 → 双维度评估
- **双维度评估**：检索指标（Recall@K/MRR/NDCG）+ 生成指标（Faithfulness/Relevance/Completeness，LLM-as-Judge）
- **多级中文分块**：标题 → 段落 → 句子 → Token 截断 + Overlap，针对中文优化
- **多策略检索**：稠密检索 / Cross-Encoder 重排序 / BM25 混合检索
- **Colab 友好**：无需本地 GPU，全部可在线运行

## 快速开始

```bash
pip install -r requirements.txt

# 1. 准备数据
# 方式A：使用内置知识库（开箱即用）
python main.py crawl --use_fallback

# 方式B：自定义数据 → 将 JSON 放入 data/articles.json

# 2. 分块 + 建索引
python main.py process --max_tokens 512

# 3. 查询
python main.py query "什么是Transformer？"

# 4. 评估
python main.py eval --n_questions 20
```

## 查询示例

```bash
python main.py query "什么是Transformer？"
python main.py query "卷积神经网络和循环神经网络有什么区别？"
python main.py query "请解释注意力机制的原理"
python main.py query --top_k 10 "深度学习面临哪些挑战？"
```

## 自定义数据格式

将数据放入 `data/articles.json`，格式如下：

```json
[
  {
    "title": "Transformer",
    "content": "文章正文内容...支持长文本",
    "topics": ["注意力机制", "自然语言处理"],
    "source": "维基百科"
  },
  {
    "title": "卷积神经网络",
    "content": "另一篇文章...",
    "topics": ["计算机视觉"],
    "source": "维基百科"
  }
]
```

- `content`：必填，文章正文
- `title`：必填，文章标题
- `topics`：可选，主题标签列表
- `source`：可选，来源名称

## 项目结构

```
rag_project/
├── main.py             # CLI 主入口 (crawl/process/query/eval/pipeline/stats)
├── config.py           # 全局配置 (API Key/分块参数/检索参数)
├── data_crawler.py     # 维基百科中文数据爬虫
├── chunker.py          # 多级递归中文分块器
├── vector_store.py     # ChromaDB 向量存储 + 多策略检索
├── rag_pipeline.py     # DeepSeek RAG 查询流水线
├── evaluator.py        # 评估模块 (检索 + LLM-as-Judge 生成评估)
├── fallback_knowledge.py  # 备选内置知识库
├── requirements.txt
├── README.md
├── data/               # 原始文章数据 (JSON)
├── chroma_db/          # ChromaDB 持久化目录
└── output/             # 评估结果和图表
```

## 评估指标

### 检索指标
| 指标 | 说明 | 理想值 |
|------|------|--------|
| Recall@K | 正确答案在前 K 个结果中的比例 | >0.8 |
| MRR | 第一个正确答案排名的倒数均值 | >0.7 |
| NDCG@K | 归一化折损累计增益 | >0.8 |

### 生成指标 (LLM-as-Judge)
| 指标 | 说明 | 评分范围 |
|------|------|----------|
| Faithfulness | 回答是否基于上下文，无幻觉 | 1-5 |
| Relevance | 回答是否直接回应问题 | 1-5 |
| Completeness | 回答是否覆盖关键信息 | 1-5 |

## 调优策略

### 提高召回率
1. **调整分块大小**: `--max_tokens 256/512/1024`（越小召回越精确，越大上下文越完整）
2. **增加 Overlap**: `--overlap_tokens 64/128`（减少边界信息丢失）
3. **更换 Embedding 模型**: `--embedding "BAAI/bge-large-zh-v1.5"`（更大模型通常更准）
4. **增加检索数量**: `--top_k 10/20`（提高召回但会增加 LLM 上下文长度）

### 提高生成质量
1. **更精确的检索** → 上下文质量更高 → 生成质量更高
2. **系统提示词优化**（修改 rag_pipeline.py 中的 SYSTEM_PROMPT）

---

## Google Colab 运行指南

### 准备工作

在本地压缩项目：

```bash
cd D:\编程\个人项目
tar -czf rag_project.tar.gz rag_project/
```

上传到 Colab（📁 文件图标 → 上传）。

### 单元格 1：解压 + 安装依赖

```python
!tar -xzf rag_project.tar.gz
%cd rag_project

!pip install -q chromadb sentence-transformers openai tiktoken rank-bm25 matplotlib scikit-learn requests

import chromadb
import sentence_transformers
from openai import OpenAI
print("✅ 依赖安装完成")
```

### 单元格 2：上传自定义数据（可选）

如果使用自己的数据，在 Colab 左侧文件管理器中将 `articles.json` 上传到 `rag_project/data/` 目录。

如果使用内置知识库，跳过此步。

### 单元格 3：分块 + 建索引

```python
%cd /content/rag_project
!python main.py process --max_tokens 512
```

### 单元格 4：交互式查询

```python
%cd /content/rag_project
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

vs = VectorStore()
rag = RAGPipeline(vs)

questions = [
    "什么是Transformer？",
    "机器学习和深度学习有什么关系？",
    "卷积神经网络主要用于什么任务？",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"问题: {q}")
    print('='*60)
    result = rag.query(q)
    print(f"回答: {result['answer'][:400]}")
    print(f"\n参考来源: {len(result['context'])} 个")
    for i, ctx in enumerate(result['context'], 1):
        title = ctx['metadata'].get('title', '未知')
        print(f"  [{i}] {title} (相关度: {ctx['score']:.3f})")
```

### 单元格 5：完整评估

```python
%cd /content/rag_project
!python main.py eval --n_questions 20 --n_eval 10
```

### 单元格 6：查看评估图表

```python
%cd /content/rag_project
from IPython.display import Image, display
display(Image(filename="output/eval_results.png"))
```

### 单元格 7（可选）：对比实验

```python
%cd /content/rag_project

# 实验1：小分块
!python main.py eval --max_tokens 256 --n_questions 15 --n_eval 8

# 实验2：大分块
!python main.py eval --max_tokens 1024 --n_questions 15 --n_eval 8
```

---

## 技术栈

- **LLM**: DeepSeek (deepseek-chat) via OpenAI-compatible API
- **Embedding**: BAAI/bge-small-zh-v1.5 (384维 中文优化)
- **Vector DB**: ChromaDB (轻量级、自动持久化)
- **Chunking**: 多级递归分块 + Overlap
- **Evaluation**: Retrieval metrics + LLM-as-Judge

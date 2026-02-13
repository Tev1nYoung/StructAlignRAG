# StructAlignRAG

StructAlignRAG 是一个零样本/免训练的多跳 RAG 原型，核心是:

- 离线构建可审计的 Evidence Memory Graph: `doc/passage/entity/capsule` 多类型节点 + typed edges + 向量索引
- 在线根据 Query DAG 在离线记忆上选择结构对齐的证据子图，再进行分步生成

运行环境与 HippoRAG 对齐:

- 环境: `conda activate hipporag`
- LLM: OpenAI-compatible API (默认 `https://integrate.api.nvidia.com/v1`)
- LLM key: 项目根目录 `llm_key.txt` (与 HippoRAG 同格式, 支持多 key, 自动轮转)
- Embedding: 默认 `facebook/contriever`
- 数据集: 直接复用 `HippoRAG/reproduce` 并复制到 `StructAlignRAG/reproduce`

## Quickstart

```powershell
cd C:\Project\StructAlignRAG
python main.py --dataset sample --force_index_from_scratch true --max_queries 1
```

常用离线参数:

```powershell
# 只用句子级 capsule 跑通(不调用离线 LLM)
python main.py --dataset sample --force_index_from_scratch true --capsule_mode sentence

# 启用缓存式 NLI edges (会增加离线 LLM 调用)
python main.py --dataset sample --force_index_from_scratch true --enable_nli_edges true
```

## Outputs (HippoRAG 风格)

输出目录:

`outputs/<dataset>/<llm_tag>_<emb_tag>/`

离线产物(核心):

- `index_meta.json`: 离线索引元信息(含 config 与产物路径)
- `offline_stats.json`: 离线各阶段耗时与 capsule 抽取统计
- `docs.jsonl`: doc 级记录(含 `doc_text`, 用于 Recall@k 严格字符串对齐)
- `passages.jsonl`: passage 切分结果(含 `sentences`, `start_char/end_char`)
- `entities.jsonl`: entity 规范化结果(aliases 聚类)
- `capsules.jsonl`: passage 级 capsule(含 `capsule_id/canonical_id/text/arguments/provenance/quality`)
- `canonical_capsules.jsonl`: canonical capsule(合并 provenance, 用于图与索引)
- `capsule_to_canonical.jsonl`: capsule_id -> canonical_id 映射
- `doc_embeddings.npy` / `passage_embeddings.npy` / `capsule_embeddings.npy` / `canonical_capsule_embeddings.npy`
- `faiss_passages.index` + `faiss_passage_ids.json`
- `faiss_capsules.index` + `faiss_capsule_ids.json`
- `graph_edges.jsonl` + `graph_adj.pkl`: typed evidence graph
- `struct_index.pkl`: 结构索引(entity->capsules, passage->capsules 等)
- `entity_alias_to_id.json`: entity 倒排(别名 -> entity_id)

在线/评测产物:

- `qa_predictions.json`
- `metrics_log.jsonl`

## Notes

- tqdm 进度条默认使用 ASCII, 避免 Windows 终端乱码。
- `enable_nli_edges=true` 时, 离线会对高相似 capsule 对进行缓存式 NLI 分类并写入 `graph_edges.jsonl`。

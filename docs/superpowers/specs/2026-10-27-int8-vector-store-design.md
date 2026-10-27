# 向量库 int8/Matryoshka 改造设计（2026-10-27）

## 目标

1. 默认 embedding 切回 `models/nomic-embed-text-v1.5-mlx-int8`（MLX 加载），默认切片 800 token。
2. 向量表按元素类型建表，默认 `int8`，体积从 2048B/行降到 512B/行。
3. nomic（及未来支持 Matryoshka 的模型）截断到 512 维（可配置 128/256/512）后**强制重新归一化**，再量化入库；查询侧同一套变换，保证两侧一致。
4. 非 Matryoshka 模型只做 FP32 → 目标元素类型的量化，入库与查询都必须执行。
5. 元素类型/维度/模型变化时**不静默删数据**，标记「待重建」并由用户显式触发重建。

## 1. 元素类型

`rag_vector_element_type ∈ {fp32, int8, fp16, bin}`，默认 `int8`。

| 值 | vec0 列 | 每行 @512 维 | 距离 | 备注 |
|---|---|---|---|---|
| `fp32` | `float[512]` | 2048B | cosine | 兼容旧库 |
| `int8` | `int8[512]` | 512B | cosine | 对称量化 `round(x×127)`，`|x|≤1` |
| `bin` | `bit[512]` | 64B | hamming | 符号位量化，LSB-first 打包，`dim % 8 == 0` |
| `fp16` | 不支持 | — | — | sqlite-vec 0.1.9 只有 float32/int8/bit，启动时报错提示改类型 |

约束：
- `int8` / `bin` 的绑定必须包 `vec_int8(?)` / `vec_bit(?)`（vec0 按 SQLite subtype
  校验元素类型），否则报类型不匹配。`bin` 不能写 `distance_metric`。
- 分数映射：cosine `1 - distance`；hamming `1 - distance / dim`（`dim` 为位数）。
- `int8` 量化必须对称、全局 scale=127：sqlite-vec 自带的 `vec_quantize_int8`
  是 `[-1,1]→[-128,127]` 的非对称映射，带 -0.5 偏移，会破坏余弦。

## 2. Matryoshka

- 配置：`rag_matryoshka ∈ {auto, on, off}`（默认 `auto`）、`rag_matryoshka_dim ∈ {128,256,512}`（默认 512）。
- `auto`：模型名含 `nomic` 时启用；`on`：任何模型都截断（`min(dim, 原生维)`）；`off`：不截断。
- 装饰器 `MatryoshkaEmbedding`：`embedDocuments` / `embedQuery` 统一
  `truncate → normalize → 返回`，`identifier` 追加 `#mrl<dim>`。
- 装饰器位于 `AdaptiveEmbeddingScheduler` 之内，入库与查询共用同一实例。

## 3. 存储与检索

- `VectorStore(... elementType:, indexFingerprint:)`；meta 记录
  `vector_element_type`、`index_fingerprint`、`embedding_dim`。
- **索引指纹** = `embedding.identifier | dimension | matryoshkaDim | elementType`。
  维度 / 元素类型 / 指纹任一不符 → `rebuildReason`，不删数据：
  - 向量通道跳过（FTS 仍可用）；
  - `ingest` 拒绝启动并提示重建；
  - `status()` 暴露 `vector_element_type` / `matryoshka_dim` /
    `vector_rebuild_required` / `rebuild_reason`。
- `RAGEngine.rebuildVectorIndex()`：按当前配置重建 `chunk_vectors`（新类型/维度），
  清 chunks/fts/documents/ANN 与 manifest、清 markdown 缓存，随后需重新同步。
- 空库首次推理发现真实维度时仍自动采纳（`didResetForDimensionChange`）。

## 4. ANN 索引适配

- `annFormatVersion = 2`；cell 行格式跟随元素类型：
  - `fp32` 库：`[rowid][scale][int8×dim]`（现状，二次量化）；
  - `int8` 库：直接复制库内码值，`scale = 1`（`dequantize = code/127`），避免二次量化；
  - `bin` 库：`[rowid][bit payload]`，每行 `8 + dim/8` 字节，汉明打分。
- 影子表快通道行宽按元素类型折算；簇心仍为 int8+scale；尾块精确补齐读取同一编码。
- 重打分读取库内编码并解码为 float，三种类型共用同一余弦口径。

## 5. 默认值与迁移

- `ragEmbeddingModel` / `localEmbeddingModel` → `nomic-embed-text-v1.5-mlx-int8`；
  `localEmbeddingModelPath` → `models/nomic-embed-text-v1.5-mlx-int8`。
- `ragChunkSize` → 800，`ragChunkOverlap` → 80。
- `migrate()`：bge 旧默认 → nomic int8；`500/50` → `800/80`；移除
  「800/40、800/80 → 500/50」迁移（否则新默认会被改回去）。

## 6. 代价

- 现有 2100 万切片库向量语义与编码全部失效，需重建（重新解析 + 重新 embedding，
  预计数十小时，可中断续跑）。
- int8 约 1% 余弦偏差；bin 只有汉明排序，召回依赖候选池与 Reranker。
- 800 token 切片使单条检索结果更长，`rag_search_max_chars_per_result` 按
  切片 token 自动放大（现有 `resultCharLimit` 逻辑）。

# Jina Embeddings v5 Text Nano 接入设计

**日期：** 2026-10-30

**目标：** 为 `jina-embeddings-v5-text-nano` 增加 MLX 4bit（默认）与 ONNX int8（可选）两种本地推理支持；入库和检索统一输出前 128 维并重新 L2 归一化，再以 int8 落盘/检索。

## 已确认决策

1. 默认 embedding 方案改为 `jina-embeddings-v5-text-nano-mlx-int4`，并打包进 `BaseAgent.app/Contents/Resources/models`。
2. `jina-embeddings-v5-text-nano-onnx-int8` 仅增加可选支持，不设为默认，不参与默认打包。
3. V5 管线强制 int8 落盘与查询；用户即使配置 fp32/bin，V5 也回落 int8。
4. 旧默认 `jina-embeddings-v3-mlx-int4` 在配置值完全等于旧默认时迁移到 V5 MLX；显式选择其他模型不迁移。
5. V5 固定输出 128 维，Matryoshka 设置不参与 V5 的维度决策。

## 非目标

- 不实现 V5 的 clustering / classification 任务适配（模型为 retrieval 单任务）。
- 不改动 reranker、LLM、OCR 的模型选择。
- 不把 ONNX 模型打进默认 app 包。

## 模型事实

两个模型目录均为 EuroBERT-210M：

- `hidden_size=768`、`num_hidden_layers=12`、`num_attention_heads=12`、`num_key_value_heads=12`
- `head_dim=64`、`intermediate_size=3072`、`vocab_size=128256`
- `max_position_embeddings=8192`、`rope_theta=1e6`
- `rms_norm_eps=1e-5`、SwiGLU、无 attention bias、无 q_norm/k_norm
- 双向 attention（仅 padding mask，无 causal mask）
- last-token pooling + L2
- 检索任务已合并 LoRA，query/doc 仅靠文本前缀区分
- MLX int4：`group_size=32`、`bits=4`、`mode=affine`；embedding 也是 packed 量化
- ONNX int8：输入 `input_ids` + `attention_mask`，输出 `embedding [B,768]`（图内 last-token + L2）

## 前缀与任务

| 内部任务 | 模型任务 | 前缀 |
| --- | --- | --- |
| `JinaV5Task.query` | `retrieval.query` | `Query: ` |
| `JinaV5Task.document` | `retrieval.passage` | `Document: ` |

- 入库分片使用 document 任务与前缀。
- 检索查询使用 query 任务与前缀。
- 前缀只在 provider 内部添加一次；`EmbeddingPrefixStrategy.apply` 保持幂等。
- `documentPrefixTokenReserve = 4`（实测前缀 3 token，留 1 个余量）。

## 向量管线

入库：

```
chunk text
→ "Document: " + text
→ tokenizer（右 padding）
→ EuroBERT forward
→ last-token pooling（768）
→ prefix(128)
→ L2 normalize
→ VectorEncoder.encodeInt8（clamp(round(x*127), -127, 127)）
→ vec_int8 入库
```

检索：

```
query text
→ "Query: " + text
→ 同一个 provider / 同一个 tokenizer / 同一套 forward
→ last-token pooling（768）
→ prefix(128)
→ L2 normalize
→ VectorEncoder.encodeInt8
→ vec_int8 最近邻检索
```

关键约束：

- 截断后必须重新归一化，不能复用截断前的范数。
- 入库与查询共用同一个 provider 实例，保证两侧变换完全一致。
- V5 provider 的 `dimension` 在加载时即为 128。
- V5 provider 实现 `EmbeddingVectorElementConstraining`，要求落盘类型 `.int8`；
  `RAGEngine.effectiveVectorElementType` 以该约束优先，索引指纹、建库、ingest
  签名与状态输出都使用生效类型。用户配置 fp32/bin 时实际仍按 int8 入库/检索。

## 组件

### `JinaV5Profile`（新增）

职责：模型识别、配置读取、任务/前缀常量、128 维重归一化工具。

- `nativeDimension = 768`
- `retrievalDimension = 128`
- `queryPrefix = "Query: "`
- `documentPrefix = "Document: "`
- `matches(_:)`：名称/标识符包含 `jina-embeddings-v5`、`jina_v5`、`jina-v5`
- `isJinaV5(directory:)`：目录名匹配，或 config 的 `base_model` 匹配，或 `model_type == "eurobert"` 且存在 `prefixes.query/doc` / `pooling` 为 last-token
- `readConfig(directory:)` 复用 JinaV3 的实现方式
- `weightsURL(in:)`：优先 `weights.safetensors`，回退 `model.safetensors`
- `truncateAndNormalize(_:dimension:)`：截前 N 维后重新 L2

### `JinaV5MLXModel`（新增）

职责：加载 `model.safetensors` 并执行 EuroBERT 前向。

- 权重键：
  - `model.embed_tokens.{weight,scales,biases}`
  - `model.layers.N.self_attn.{q_proj,k_proj,v_proj,o_proj}.{weight,scales,biases}`
  - `model.layers.N.mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}`
  - `model.layers.N.input_layernorm.weight`
  - `model.layers.N.post_attention_layernorm.weight`
  - `model.norm.weight`
- embedding：先按 token ids gather packed 权重/scales/biases，再 `dequantized`，避免整表反量化。
- Linear：`quantizedMM(x, weight, scales:biases:transpose:true, groupSize:32, bits:4, mode:.affine)`。
- Attention：q/k/v 线性 → reshape `[B,S,H,64]` → RoPE（`traditional=false`、`base=1e6`）→ 转置 `[B,H,S,64]` → 加性 padding mask → softmax → o_proj。
- MLP：`down(silu(gate(x)) * up(x))`。
- 归一化：`rmsNorm(x, weight:, eps: 1e-5)`。
- Pooling：按 attention mask 求每条序列最后一个有效 token 的 hidden。
- 输出：先取前 128 维，再 L2；返回 `[[Float]]`。
- 权重加载失败、缺少键、shape 不符时抛 `LocalModelError.runtime/unsupportedArchitecture`。

### `JinaV5MLXEmbeddingProvider`（新增）

职责：把模型包装成 `EmbeddingProvider + EmbeddingTokenCounting`。

- `identifier = "local:mlx-embedding:\(modelName)"`
- `dimension = 128`
- `maxConcurrentRuns = 1`
- `requiredVectorElementType = .int8`（实现 `EmbeddingVectorElementConstraining`）
- `embedDocuments`：加 `Document: ` 后调用 model。
- `embedQuery`：加 `Query: ` 后调用 model。
- 懒加载 model + tokenizer；`unload()` 释放。
- `maxContentTokens`：窗口（现有 `EmbeddingInputLimits.cappedWindow`，1024）减特殊 token 与前缀预留。
- `makeTokenCounter` 走 `TokenizerCounterCache`。

### ONNX 扩展（修改 `ONNXEmbeddingProvider`）

- 新增 `isJinaV5` 检测。
- V5 时：
  - `prefixStrategy = EmbeddingPrefixStrategy(family: .jinaV5)`
  - `storedDimension = 128`
  - `pool` 对 2D 输出截前 128 维并重新归一化
  - 若输出 3D，按 attention mask 做 last-token pooling，再截断/归一化
  - 不要求 `task_id` 输入（V5 ONNX 图没有该输入）
  - 实现 `EmbeddingVectorElementConstraining`，要求 `.int8`
- `contentTokenLimit` 对 V5 使用 `.jinaV5` 的前缀预留。

### 前缀策略（修改 `EmbeddingPrefixStrategy`）

- 新增 `EmbeddingModelFamily.jinaV5`
- `prefix(for:)`：document `"Document: "`，query `"Query: "`
- `documentPrefixTokenReserve = 4`
- `detectFamily`：`jina` + `v5` → `.jinaV5`；其余 jina → `.jina`，保持 v3 的 `passage:/query:` 不变。

### Tokenizer 回退（修改 `LocalTokenizer`）

两个 V5 目录都只有 `tokenizer.json`，没有 `tokenizer_config.json`。

- `tokenizer_config.json` 存在：保持现有 `AutoTokenizer.from(modelFolder:)` 路径。
- 缺失时：
  - 用 `HubApi.shared.configuration(fileURL:)` 读取 `tokenizer.json`
  - 合成最小配置：`tokenizer_class = "PreTrainedTokenizer"`、`model_max_length = 8192`
  - `AutoTokenizer.from(tokenizerConfig:tokenizerData:strict:false)`
  - pad 从 `tokenizer.json` 的 `padding.pad_id` 或 `padding.pad_token` 读取；回退 eos/0
- 不影响现有 LLM tokenizer。

### Matryoshka（修改）

- `MatryoshkaEmbedding.wrap` 对 V5 provider 直接返回 base，避免 `on + 64` 二次截断。
- `MatryoshkaMode.auto` 的显示文案增加 jina-v5 固定 128 维说明（无功能变化）。

### 配置、工厂、注册表、UI、打包

- `AgentConfig` 默认：
  - `ragEmbeddingModel = "jina-embeddings-v5-text-nano-mlx-int4"`
  - `localEmbeddingModel = "jina-embeddings-v5-text-nano-mlx-int4"`
  - `localEmbeddingModelPath = "models/jina-embeddings-v5-text-nano-mlx-int4"`
  - `localEmbeddingBackend = "mlx"`
  - `ragMatryoshkaDim = 128`
- `migrate()`：仅当 model/path 完全等于旧默认 `jina-embeddings-v3-mlx-int4` 时改为 V5。
- `LocalModelRegistry.defaultName(for: .embedding)` 改 V5 MLX。
- `LocalModelRegistry.applicationModelsDirectory()` 增加 app bundle `Contents/Resources/models`。
- `LocalModelFactory.makeEmbedding`：MLX 且 `JinaV5Profile.isJinaV5(directory:)` → `JinaV5MLXEmbeddingProvider`；ONNX 继续通用 provider。
- `SettingsView`：默认路径改 V5；V5 时显示“固定 128 维 + int8（Matryoshka 不生效）”，元素类型选择器禁用，保存时写回 int8。
- `RAGEngine`：新增 `effectiveVectorElementType`（provider 约束优先，否则用配置）；索引指纹、`VectorStore` 初始化、ingest 签名与 `status` 均读取生效类型。
- `package-app.sh`：复制 V5 MLX 的 `config.json`、`model.safetensors` / `weights.safetensors`、`tokenizer.json`、`tokenizer_config.json`、`README.md` 到 `Contents/Resources/models/...`。

## 错误处理

- 目录不存在/缺权重/缺 tokenizer：沿用 `LocalModelError`，错误信息包含目录路径。
- MLX 权重缺键：启动时立即抛错，不静默降级到通用 MLXEmbeddingProvider。
- ONNX 图缺输出/输入：沿用 `ONNXRuntimeError`。
- 空文本：tokenizer 保留一个 pad 位；pooling 仍能返回向量。

## 测试

1. `EmbeddingPrefixStrategyTests`：jina-v5 前缀、v3 不变、reserve。
2. `JinaV5ProfileTests`：名称/config 识别、128 维截断重归一化、任务映射。
3. `MatryoshkaEmbeddingTests`：V5 不被二次截断。
4. `LocalTokenizerFallbackTests`：缺 `tokenizer_config.json` 时能加载 V5 tokenizer（模型存在才跑）。
5. `JinaV5MLXEmbeddingE2ETests`（模型存在才跑）：
   - 维度 128、范数 1
   - query/doc 前缀生效
   - 相关文档排序正确
   - 单条与批量一致
   - 与 Python 参考余弦对齐
6. `JinaV5ONNXEmbeddingE2ETests`（ONNX 目录存在才跑）：同样断言。
7. RAG 全链路：V5 入库/检索，int8 blob 长度为 128，Recall@1 正确。
8. 强制 int8：配置 fp32 时 `effectiveVectorElementType`、指纹与 `status` 仍为 int8。
9. 全量 `swift test` 回归。

## 风险

- MLX 前向与 Python 参考存在数值误差；以余弦排序和范数为主断言，不逐元素比对。
- 打包后 bundle 内模型路径解析需覆盖 `Contents/Resources/models` 与裸可执行文件两种布局。
- 旧向量库维度从 256 变为 128，升级后需用户重建；UI 已有重建提示。

# Embedding 输入前缀统一策略

日期：2026-10-28

## 背景

ONNX、MLX、GGUF 三个 embedding provider 各自判断是否加任务前缀：

- ONNX 无条件加 nomic 的 `search_document: ` / `search_query: `，非 nomic 模型会被错误加前缀；
- MLX 只识别 nomic 与 bge-zh；
- GGUF 只识别 nomic。

模型族与后端组合不同会导致同一模型的输入格式不一致，直接影响向量检索质量。

## 决策

新增共享的 `EmbeddingPrefixStrategy`（`Sources/BaseAgentCore/Local/EmbeddingPrefixStrategy.swift`），
三个 provider 统一委托给它。调用方（RAGEngine、记忆、技能检索）保持不变，仍只调用
`embedDocuments` / `embedQuery`。

## 前缀映射

| 模型族 | document | query | clustering | classification |
| --- | --- | --- | --- | --- |
| nomic | `search_document: ` | `search_query: ` | 无 | 无 |
| bge-zh | 无 | `为这个句子生成表示以用于检索相关文章：` | 无 | 无 |
| jina | `passage: ` | `query: ` | `clustering: ` | `classification: ` |
| e5 | `passage: ` | `query: ` | 无 | 无 |
| 其他 | 无 | 无 | 无 | 无 |

要求：

- 所有带冒号的前缀，冒号后保留一个空格；
- bge 指令只对 bge-zh 生效，bge-en、bge-m3 等不加前缀；
- 前缀拼接幂等，文本已带目标前缀时不重复添加；
- `clustering` / `classification` 只作为策略映射存在，本次不新增任务型 embedding API。

## 模型族识别

- 同时检查模型名与模型路径的最后一段（目录名或文件名），避免用户名/上级目录误匹配；
- 按非字母数字分隔后做 token 匹配：`nomic`、`bge`、`jina`、`e5`；
- bge 只有在同时出现 `zh` token 时才视为 bge-zh；
- `e5` 必须作为独立 token 匹配，避免随机路径字符串误判；
- 未知模型一律不加前缀。

## 集成

- `ONNXEmbeddingProvider`：移除无条件 nomic 前缀，改为按策略；
- `MLXEmbeddingProvider`：移除 `usesNomicPrefix` / `usesBGEQueryInstruction`；
- `LlamaCppEmbeddingProvider`：移除只认 nomic 的判断；
- 切片 token 预算按 document 前缀预留：nomic 8 token、jina/E5 4 token、bge-zh/其他 0 token
  （特殊 token 仍统一扣 2）。GGUF 的实例 `maxContentTokens` 同步扣除预留。

## 测试

新增 `EmbeddingPrefixStrategyTests`：

- 四个模型族的前缀映射与幂等；
- bge-en、bge-m3 不加任何前缀；
- 路径含 `zhang` 等字符串不会误判为 bge-zh；
- `3e5f` 不会误判为 E5，`e5-large-v2` 正常识别；
- 模型名不含族名时，可通过模型路径识别；
- 未知模型不加前缀。

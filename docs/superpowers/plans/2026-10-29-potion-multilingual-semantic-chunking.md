# potion-multilingual-128M 语义分片接入 Implementation Plan

> 本方案替换原 `2026-10-29-bge-small-zh-semantic-chunking.md`（旧 bge 资源、代码、测试与基准模型已清理）。

**Goal:** 用内置 `potion-multilingual-128M` int4 ONNX（纯 CPU）替换 `bge-small-zh-v1.5` int8，解决语义切片长句超过 512 token 被截断的问题，同时提升推理速度，并随 App 资源固化分发。

**Architecture:** SwiftPM `.copy` 资源携带 int4 ONNX（约 99 MB：model 80 MB + tokenizer 18.6 MB + config）；新增 `PotionSentenceSimilarity`，用 `Tokenizers`（Unigram）同步分词、`OrtSession`（CPU EP）推理、模型内建 mean-pool + L2 归一化后计算相邻句余弦。`SemanticChunker` / `TextChunker` 的默认相似度闭包切换到该实现，失败时仍回退词法相似度。`pipeline_version` 升到 11。

**Tech Stack:** Swift 6（v5 语言模式）、XCTest、swift-transformers `Tokenizers` + `Hub`、onnxruntime-swift-package-manager、SwiftPM resources。

---

## 文件结构

- 新建：`Sources/BaseAgentCore/RAG/Chunking/PotionSentenceSimilarity.swift` — 资源定位、Unigram 分词、分桶推理、相邻句余弦。
- 新建：`Sources/BaseAgentCore/Resources/potion-multilingual-128M-int4/` — `model.onnx`、`tokenizer.json`、`config.json`、`LICENSE`、`MODEL_INFO.md`。
- 修改：`Package.swift` — BaseAgentCore 增加 `Hub` product 依赖，资源改为 `.copy("Resources/potion-multilingual-128M-int4")`。
- 修改：`Sources/BaseAgentCore/RAG/Chunking/SemanticChunker.swift` — 默认相似度闭包、自适应阈值注释与分位下标。
- 修改：`Sources/BaseAgentCore/RAG/Chunking/TextChunker.swift` — 默认相似度闭包、过时窗口注释。
- 修改：`Sources/BaseAgentCore/RAG/RAGEngine.swift` — `pipeline_version` 10 → 11。
- 修改：`Sources/BaseAgentCore/RAG/Chunking/{ChunkingProtocol,EmbeddingTokenEstimator}.swift`、`Sources/BaseAgentApp/SettingsView.swift` — 清理 bge 窗口/内存相关过时文案。
- 新建：`Tests/BaseAgentCoreTests/PotionSentenceSimilarityTests.swift` — 资源、分词、向量、长输入、语义区分度、性能、批处理正确性。
- 新建：`Tests/BaseAgentCoreTests/Fixtures/potion-multilingual-golden.json` — Python（`tokenizers` 不截断 + `onnxruntime`）生成的 token ids + 256 维单位向量。
- 修改：`Tests/BaseAgentCoreTests/EmbeddingPathBenchTests.swift` — 基准探针改用 `PotionSentenceSimilarity`。
- 修改：`Tests/BaseAgentCoreTests/ChunkingTests.swift` — 保留真实模型三主题集成测试。
- 删除：`BGESentenceSimilarity.swift`、`BGESentenceSimilarityTests.swift`、`Fixtures/bge-small-zh-golden.json`、`Resources/bge-small-zh-v1.5-int8/`、`models/bge-small-zh-v1.5-mlx/`。

---

## Task 1: 模型可用性验证（Python）

- [x] int4 ONNX 加载成功，输出 `float32[batch, 256]`，L2 norm = 1。
- [x] 中文相关句余弦约 0.52，无关句约 0.037（区分度低于 bge 的 0.72 / 0.38，符合预期）。
- [x] 关闭官方 `max_length=512` 截断后，6000 / 10000 / 30000 / 100000 token 输入均可推理，输出仍为单位向量；纯 CPU 耗时约 0.013 / 0.020 / 0.086 / 0.300 s。
- [x] int4 相对 fp32 余弦均值 0.998110、最小 0.996149（出自转换侧 `README` / `verification_report`）。

## Task 2: 资源固化

- [x] 复制 `models/potion-multilingual-128M-onnx/int4/{model.onnx,tokenizer.json,config.json}` 到 `Sources/BaseAgentCore/Resources/potion-multilingual-128M-int4/`。
- [x] 补 `LICENSE`（MIT，MinishLab）与 `MODEL_INFO.md`（来源、SHA-256、量化参数、维度、无 512 限制、实测数据）。
- [x] `config.json` 的 `source_model` 由本地绝对路径改为 `minishlab/potion-multilingual-128M`，避免把开发机路径打进 App。
- [x] `Package.swift` 用 `.copy` 声明资源，`Bundle.module` 内可直接定位目录。

## Task 3: 分词与推理

- [x] 用 `Config` + `AutoTokenizer.from(tokenizerConfig:tokenizerData:strict:)` 直接构造 tokenizer；`tokenizer_class = "XLMRobertaTokenizer"` 在库内映射到 `UnigramTokenizer`（目录无 `tokenizer_config.json`，不能走 `from(modelFolder:)`）。
- [x] `addSpecialTokens: false`，丢弃 `unk_id = 1`（与官方 model2vec 一致）；空句/纯空白直接返回 nil，避免 metaspace 产出 `▁` 被当作有效句。
- [x] 不采用 `config.max_length = 512` 截断；仅保留 8192 token 防御上限。
- [x] 按 token 数排序后切子批（≤32 句且 ≤32768 token/批），推理后按原下标放回，避免长句 padding 放大内存。
- [x] 输入只接受 `input_ids` / `attention_mask`，padding token 0 由 mask 排除；输出兼容 `[B,H]` 与 `[B,T,H]`。

## Task 4: 测试与阈值

- [x] token ids 与 HF `tokenizers` golden 逐句一致；golden 含 1 句 > 512 token，Swift 不截断。
- [x] 向量与 Python golden 同向（逐句余弦 > 0.99），句对余弦误差 < 0.01；维度 256、norm ≈ 1。
- [x] 中文相关句比无关句高 0.2 以上；空句返回 nil。
- [x] 200 句纯 CPU 推理 < 5 s（模型已加载时实测约 0.08 s）；长短混合批相同文本余弦 > 0.9999，不串位。
- [x] 真实模型集成：三主题 9 句切 3 段。
- [x] 自适应阈值取样本 25 分位时下标改为 `n-1`（原 `n` 取到第 3 低值，potion 下会把「鱼类」主题内第 2/3 句误切）。

## 验收清单

- [x] 9 个 `PotionSentenceSimilarityTests` 全部通过。
- [x] 15 个 `ChunkingTests` 全部通过（含内置模型集成）。
- [x] `EmbeddingPathBenchTests` 不再引用 `BGESentenceSimilarity`。
- [x] 仓库内不再有 `BGESentenceSimilarity`、`bge-small-zh-golden.json`、`Resources/bge-small-zh-v1.5-int8/` 引用或文件。
- [x] 本地旧模型目录 `models/bge-small-zh-v1.5-mlx/` 已删除；依赖它的基准测试按既有 `XCTSkip` 跳过。
- [x] `swift test` 全量通过（372 项 / 25 skip / 0 failure）；`swift build` 通过。

## 执行中发现的偏差

1. **首次加载耗时**：tokenizer 构造约 5.4 s（500353 词表 Unigram 解析），ONNX session 约 0.05 s；只发生一次（进程内缓存），之后 200 句推理约 0.08 s。若后续要优化启动体验，可考虑预热或在后台加载。
2. **阈值分位**：potion 余弦整体低于 bge，固定 0.55 不再适用；自适应阈值仍按低分位工作，但 25 分位下标需按 `n-1` 取，避免多切一个主题内边界。
3. **中文区分度**：potion 相关/无关句差距约 0.48，高于 bge 的 0.34；但余弦绝对值整体更低，因此阈值逻辑依赖低分位而非固定 0.55。

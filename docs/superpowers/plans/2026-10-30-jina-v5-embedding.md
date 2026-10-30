# Jina Embeddings v5 Text Nano Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 增加 `jina-embeddings-v5-text-nano` 的 MLX 4bit（默认）与 ONNX int8（可选）支持，统一 128 维截断 + L2 重归一化 + int8 入库/检索。

**Architecture:** 新增共享 `JinaV5Profile` 负责模型识别、前缀/任务与 128 维变换；MLX 与 ONNX 各自实现推理 provider。V5 provider 直接输出 128 维单位向量，Storage 层继续复用 `VectorEncoder.encodeInt8`。默认模型切到 V5 MLX，ONNX 仅作为显式配置的可选后端。

**Tech Stack:** Swift 6、SwiftPM、XCTest、MLX Swift、onnxruntime、swift-transformers、SQLite vec_int8。

---

### Task 1: 共享 V5 识别、前缀与 128 维变换

**Files:**
- Create: `Sources/BaseAgentCore/Local/JinaV5Support.swift`
- Modify: `Sources/BaseAgentCore/Local/EmbeddingPrefixStrategy.swift`
- Test: `Tests/BaseAgentCoreTests/JinaV5SupportTests.swift`
- Test: `Tests/BaseAgentCoreTests/EmbeddingPrefixStrategyTests.swift`

- [ ] **Step 1: 写失败测试**

```swift
func testJinaV5PrefixStrategyUsesCapitalizedTaskLabels() {
    let strategy = EmbeddingPrefixStrategy(family: .jinaV5)
    XCTAssertEqual(strategy.prefix(for: .retrievalDocument), "Document: ")
    XCTAssertEqual(strategy.prefix(for: .retrievalQuery), "Query: ")
    XCTAssertEqual(strategy.documentPrefixTokenReserve, 4)
}

func testDetectFamilyDistinguishesV5FromV3() {
    XCTAssertEqual(
        EmbeddingPrefixStrategy.detectFamily(
            modelName: "jina-embeddings-v5-text-nano-mlx-int4"),
        .jinaV5)
    XCTAssertEqual(
        EmbeddingPrefixStrategy.detectFamily(
            modelName: "jina-embeddings-v3-mlx-int4"),
        .jina)
}

func testJinaV5ProfileTruncatesAndRenormalizes() {
    let input = (0..<768).map { Float($0 + 1) }
    let output = JinaV5Profile.truncateAndNormalize(input)
    XCTAssertEqual(output.count, 128)
    let norm = sqrt(output.reduce(Float(0)) { $0 + $1 * $1 })
    XCTAssertEqual(norm, 1.0, accuracy: 1e-5)
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `swift test --filter 'JinaV5SupportTests|EmbeddingPrefixStrategyTests'`
Expected: 编译失败，`JinaV5Profile` / `.jinaV5` 不存在。

- [ ] **Step 3: 实现最小代码**

`JinaV5Support.swift`：

```swift
import Foundation

enum JinaV5Profile {
    static let nativeDimension = 768
    static let retrievalDimension = 128
    static let queryPrefix = "Query: "
    static let documentPrefix = "Document: "

    static func matches(_ text: String) -> Bool {
        let lower = text.lowercased()
        return lower.contains("jina-embeddings-v5")
            || lower.contains("jina_v5")
            || lower.contains("jina-v5")
    }

    static func isJinaV5(directory: URL) -> Bool {
        if matches(directory.lastPathComponent) { return true }
        guard let config = readConfig(directory: directory) else { return false }
        if let base = config["base_model"]?.stringValue, matches(base) { return true }
        if let type = config["model_type"]?.stringValue, matches(type) { return true }
        let prefixes = config["prefixes"]?.objectValue
        if type?.lowercased() == "eurobert",
           prefixes?["query"]?.stringValue == queryPrefix,
           prefixes?["doc"]?.stringValue == documentPrefix {
            return true
        }
        return false
    }

    static func readConfig(directory: URL) -> [String: JSONValue]? {
        let url = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
              let text = String(data: data, encoding: .utf8),
              let object = JSONValue.parse(text)?.objectValue else { return nil }
        return object
    }

    static func weightsURL(in directory: URL) -> URL? {
        let url = directory.appendingPathComponent("model.safetensors")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    static func truncateAndNormalize(_ vector: [Float],
                                     dimension: Int = retrievalDimension) -> [Float] {
        var output = Array(vector.prefix(max(1, dimension)))
        var norm: Float = 0
        for value in output { norm += value * value }
        norm = norm.squareRoot()
        guard norm > 0 else { return output }
        for index in output.indices { output[index] /= norm }
        return output
    }
}
```

`EmbeddingPrefixStrategy` 增加 `.jinaV5` 分支与识别。

- [ ] **Step 4: 运行测试确认通过**

Run: `swift test --filter 'JinaV5SupportTests|EmbeddingPrefixStrategyTests'`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add Sources/BaseAgentCore/Local/JinaV5Support.swift \
        Sources/BaseAgentCore/Local/EmbeddingPrefixStrategy.swift \
        Tests/BaseAgentCoreTests/JinaV5SupportTests.swift \
        Tests/BaseAgentCoreTests/EmbeddingPrefixStrategyTests.swift
git commit -m "feat(embedding): add jina-v5 profile and task prefixes"
```

### Task 2: Matryoshka 跳过 V5、配置默认值与迁移

**Files:**
- Modify: `Sources/BaseAgentCore/RAG/MatryoshkaEmbedding.swift`
- Modify: `Sources/BaseAgentCore/Core/Config.swift`
- Modify: `Sources/BaseAgentCore/Local/ModelRegistry.swift`
- Modify: `Sources/BaseAgentApp/SettingsView.swift`
- Test: `Tests/BaseAgentCoreTests/MatryoshkaEmbeddingTests.swift`
- Test: `Tests/BaseAgentCoreTests/ParserTests.swift`

- [ ] **Step 1: 写失败测试**

```swift
func testJinaV5ProviderIsNotWrappedByMatryoshka() {
    let base = StubEmbeddingProvider(identifier: "local:mlx-embedding:jina-embeddings-v5-text-nano-mlx-int4",
                                     dimension: 128)
    let wrapped = MatryoshkaEmbedding.wrap(base, mode: .on, dimension: 64)
    XCTAssertEqual(wrapped.dimension, 128)
    XCTAssertTrue(wrapped === base)
}
```

配置测试断言默认模型/路径为 V5、`ragMatryoshkaDim == 128`，迁移旧 v3 默认值后为 V5。

- [ ] **Step 2: 运行测试确认失败**

Run: `swift test --filter 'MatryoshkaEmbeddingTests|ParserTests'`
Expected: FAIL（默认值仍是 v3，wrap 仍会截断）。

- [ ] **Step 3: 实现**

- `MatryoshkaEmbedding.wrap` 开头：`if JinaV5Profile.matches(base.identifier) { return base }`
- `MatryoshkaMode.auto` 显示文案补充“jina-v5 固定 128 维”。
- `AgentConfig` 默认值改为 V5 MLX，`ragMatryoshkaDim = 128`。
- `migrate()` 只迁移完全等于旧默认的 model/path。
- `ModelRegistry.defaultName(for: .embedding)` 改 V5。
- `SettingsView.embeddingPath` 改 V5，V5 时显示固定 128 维提示。

- [ ] **Step 4: 运行测试**

Run: `swift test --filter 'MatryoshkaEmbeddingTests|ParserTests'`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add Sources/BaseAgentCore/RAG/MatryoshkaEmbedding.swift \
        Sources/BaseAgentCore/Core/Config.swift \
        Sources/BaseAgentCore/Local/ModelRegistry.swift \
        Sources/BaseAgentApp/SettingsView.swift \
        Tests/BaseAgentCoreTests/MatryoshkaEmbeddingTests.swift \
        Tests/BaseAgentCoreTests/ParserTests.swift
git commit -m "feat(embedding): default to jina-v5 mlx and keep 128 dimensions"
```

### Task 3: Tokenizer 缺失 `tokenizer_config.json` 回退

**Files:**
- Modify: `Sources/BaseAgentCore/Local/TokenizerSupport.swift`
- Test: `Tests/BaseAgentCoreTests/TokenizerFallbackTests.swift`

- [ ] **Step 1: 写失败测试**

使用真实 V5 tokenizer 目录；目录不存在则 `XCTSkip`。

```swift
func testLoadsTokenizerWithoutTokenizerConfig() async throws {
    let directory = URL(fileURLWithPath: "models/jina-embeddings-v5-text-nano-mlx-int4")
    try XCTSkipUnless(FileManager.default.fileExists(
        atPath: directory.appendingPathComponent("tokenizer.json").path))
    let tokenizer = try await LocalTokenizer.load(directory: directory)
    XCTAssertEqual(tokenizer.encode("Document: hello").isEmpty, false)
    XCTAssertGreaterThan(tokenizer.padTokenId, 0)
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `swift test --filter TokenizerFallbackTests`
Expected: FAIL，抛 `missingTokenizer(file: "tokenizer_config.json")`。

- [ ] **Step 3: 实现**

- `load` 在 `tokenizer_config.json` 缺失时不再抛错。
- 用 `HubApi.shared.configuration(fileURL:)` 读取 `tokenizer.json`。
- 合成 `Config(["tokenizer_class": "PreTrainedTokenizer", "model_max_length": 8192])`。
- 调 `AutoTokenizer.from(tokenizerConfig:tokenizerData:strict:false)`。
- pad 从 tokenizer.json 的 `padding.pad_id` / `padding.pad_token` 读取。

- [ ] **Step 4: 运行测试**

Run: `swift test --filter TokenizerFallbackTests`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add Sources/BaseAgentCore/Local/TokenizerSupport.swift \
        Tests/BaseAgentCoreTests/TokenizerFallbackTests.swift
git commit -m "fix(tokenizer): load jina-v5 tokenizer without tokenizer_config"
```

### Task 4: MLX 模型与 provider

**Files:**
- Create: `Sources/BaseAgentCore/Local/JinaV5MLXModel.swift`
- Create: `Sources/BaseAgentCore/Local/JinaV5MLXEmbeddingProvider.swift`
- Test: `Tests/BaseAgentCoreTests/JinaV5MLXEmbeddingE2ETests.swift`

- [ ] **Step 1: 写失败 E2E 测试**

```swift
func testEmbeds128DimensionalNormalizedVectorsAndRanksRelevantDocumentFirst() async throws {
    let directory = URL(fileURLWithPath: "models/jina-embeddings-v5-text-nano-mlx-int4")
    try XCTSkipUnless(JinaV5Profile.isJinaV5(directory: directory))
    let provider = JinaV5MLXEmbeddingProvider(
        directory: directory, modelName: "jina-embeddings-v5-text-nano-mlx-int4")
    let query = try await provider.embedQuery("苹果发布了新手机")
    XCTAssertEqual(query.count, 128)
    XCTAssertEqual(sqrt(query.reduce(0) { $0 + $1 * $1 }), 1.0, accuracy: 1e-3)
    let docs = try await provider.embedDocuments([
        "苹果公司推出了新款智能手机。",
        "机器学习通过数据训练模型，让计算机自动改进任务表现。",
        "我想吃一个红苹果。",
    ])
    XCTAssertEqual(docs.map(\.count), [128, 128, 128])
    let scores = docs.map { cosine(query, $0) }
    XCTAssertEqual(scores[0], 0.829, accuracy: 0.08)
    XCTAssertGreaterThan(scores[0], scores[1])
    XCTAssertGreaterThan(scores[0], scores[2])
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `swift test --filter JinaV5MLXEmbeddingE2ETests`
Expected: 编译失败，类型不存在。

- [ ] **Step 3: 实现 `JinaV5MLXModel`**

按 spec 的权重键与前向步骤实现：

```swift
final class JinaV5MLXModel: @unchecked Sendable {
    // QuantizedLinear + Layer + config
    func encode(inputIDs: [[Int]], attentionMask: [[Int]]) -> [[Float]]
}
```

关键点：

- embedding gather 后 `dequantized`；
- `quantizedMM` 参数 `groupSize: 32, bits: 4, mode: .affine`；
- `MLXFast.RoPE(... traditional: false, base: 1e6, scale: 1, offset: 0)`；
- 加性 padding mask `-1e9`，softmax 后乘 V；
- RMSNorm `rmsNorm(..., eps: 1e-5)`；
- last-token 索引 `attentionMask.sum - 1`；
- 取前 128 维后 `l2Normalized`。

- [ ] **Step 4: 实现 provider**

```swift
final class JinaV5MLXEmbeddingProvider: EmbeddingProvider,
                                       EmbeddingTokenCounting, @unchecked Sendable {
    let identifier: String
    let modelName: String
    let directory: URL
    var dimension: Int { JinaV5Profile.retrievalDimension }
    var maxConcurrentRuns: Int { 1 }
    // embedDocuments / embedQuery / unload / maxContentTokens / makeTokenCounter
}
```

- [ ] **Step 5: 运行 E2E 与单测**

Run: `swift test --filter 'JinaV5MLXEmbeddingE2ETests|JinaV5SupportTests'`
Expected: PASS（模型存在时）。

- [ ] **Step 6: 提交**

```bash
git add Sources/BaseAgentCore/Local/JinaV5MLXModel.swift \
        Sources/BaseAgentCore/Local/JinaV5MLXEmbeddingProvider.swift \
        Tests/BaseAgentCoreTests/JinaV5MLXEmbeddingE2ETests.swift
git commit -m "feat(embedding): add jina-v5 mlx model and provider"
```

### Task 5: ONNX V5 支持

**Files:**
- Modify: `Sources/BaseAgentCore/Local/ONNXEmbeddingProvider.swift`
- Test: `Tests/BaseAgentCoreTests/JinaV5ONNXEmbeddingE2ETests.swift`

- [ ] **Step 1: 写失败 E2E 测试**

默认路径 `models/jina-embeddings-v5-text-nano-onnx-int8`；也支持环境变量 `JINA_V5_ONNX_DIR`。

```swift
func testOnnxEmbeds128DimensionalNormalizedVectors() async throws { ... }
```

- [ ] **Step 2: 运行测试确认失败**

Run: `swift test --filter JinaV5ONNXEmbeddingE2ETests`
Expected: 模型目录不存在时 skip；存在时维度 768 而非 128。

- [ ] **Step 3: 实现**

- `isJinaV5 = JinaV5Profile.isJinaV5(directory:)`
- 前缀策略优先 `.jinaV5`
- `storedDimension = 128`
- `pool` 2D：`JinaV5Profile.truncateAndNormalize`
- `pool` 3D：last-token pooling + 截断
- `contentTokenLimit` 识别 V5
- 保留 v3 的 `task_id` 行为

- [ ] **Step 4: 运行测试**

Run: `swift test --filter 'JinaV5ONNXEmbeddingE2ETests|JinaV3SupportTests'`
Expected: PASS（V3 行为不回归）。

- [ ] **Step 5: 提交**

```bash
git add Sources/BaseAgentCore/Local/ONNXEmbeddingProvider.swift \
        Tests/BaseAgentCoreTests/JinaV5ONNXEmbeddingE2ETests.swift
git commit -m "feat(embedding): support jina-v5 onnx embeddings"
```

### Task 6: 工厂、bundle 模型路径与打包

**Files:**
- Modify: `Sources/BaseAgentCore/Local/LocalModelFactory.swift`
- Modify: `Sources/BaseAgentCore/Local/ModelRegistry.swift`
- Modify: `scripts/package-app.sh`
- Test: `Tests/BaseAgentCoreTests/LocalModelFactoryTests.swift`

- [ ] **Step 1: 写失败测试**

断言 MLX + V5 目录返回 `dimension == 128` 且不是 Failing provider；bundle 资源目录能被 `applicationModelsDirectory()` 找到（用临时目录模拟）。

- [ ] **Step 2: 运行测试确认失败**

Run: `swift test --filter LocalModelFactoryTests`
Expected: FAIL。

- [ ] **Step 3: 实现**

- `makeEmbedding` MLX 分支：`JinaV5Profile.isJinaV5(directory:)` → `JinaV5MLXEmbeddingProvider`。
- 默认 MLX 模型名改为 V5。
- `applicationModelsDirectory()` 增加 `Bundle.main.resourceURL/models` 与 bundle `Contents/Resources/models`。
- `package-app.sh` 复制 V5 MLX 必要文件。

- [ ] **Step 4: 运行测试**

Run: `swift test --filter LocalModelFactoryTests`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add Sources/BaseAgentCore/Local/LocalModelFactory.swift \
        Sources/BaseAgentCore/Local/ModelRegistry.swift \
        scripts/package-app.sh \
        Tests/BaseAgentCoreTests/LocalModelFactoryTests.swift
git commit -m "feat(embedding): wire jina-v5 factory, bundle path and packaging"
```

### Task 7: RAG int8 全链路与全量回归

**Files:**
- Test: `Tests/BaseAgentCoreTests/JinaV5RAGIngestE2ETests.swift`
- Modify: 如测试暴露问题，回到对应实现文件。

- [ ] **Step 1: 写失败测试**

使用真实 V5 MLX provider，临时知识库，入库 3 个文档，断言：

- 向量 blob 长度 128（int8）
- 查询向量 128
- 相关文档排第一
- `rag_search` 返回结果

- [ ] **Step 2: 运行测试**

Run: `swift test --filter JinaV5RAGIngestE2ETests`
Expected: PASS（修复前可能因维度/量化失败）。

- [ ] **Step 3: 全量回归**

Run: `swift test`
Expected: 全部通过，无 failures。

- [ ] **Step 4: 提交**

```bash
git add Tests/BaseAgentCoreTests/JinaV5RAGIngestE2ETests.swift
git commit -m "test(embedding): cover jina-v5 int8 ingest and retrieval"
```

### Task 8: 文档与最终检查

**Files:**
- Modify: `docs/superpowers/specs/2026-10-30-jina-v5-embedding-design.md`（如实现有偏差）
- Modify: `README.md` 或相关文档（如存在 embedding 配置说明）

- [ ] **Step 1: 检查 spec 与实现一致**

逐项核对默认模型、前缀、128 维重归一化、int8、ONNX 可选、打包路径。

- [ ] **Step 2: 运行格式/构建检查**

Run: `swift build && swift test`
Expected: 构建成功，测试全过。

- [ ] **Step 3: 提交文档**

```bash
git add docs/superpowers/specs/2026-10-30-jina-v5-embedding-design.md README.md
git commit -m "docs: document jina-v5 embedding integration"
```

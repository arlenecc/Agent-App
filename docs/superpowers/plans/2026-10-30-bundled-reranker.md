# 内置 jina-reranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `jina-reranker-v2-base-multilingual-onnx` 打包进 App，并让默认配置无需填写路径即可使用内置 Reranker。

**Architecture:** 复用现有 `Contents/Resources/models` 解析优先级与 `LocalModelFactory` 默认名回落；只新增打包清单、回归测试、UI 提示与文档，不引入新配置项。

**Tech Stack:** Swift 6、XCTest、ONNX Runtime、Shell（`scripts/package-app.sh`）。

---

### Task 1: 回归测试（打包清单 RED + bundle 解析 + 真实模型 E2E）

**Files:**
- Create: `Tests/BaseAgentCoreTests/BundledRerankerTests.swift`
- Create: `Tests/BaseAgentCoreTests/JinaRerankerE2ETests.swift`

- [ ] **Step 1: 写打包清单与默认解析失败/回归测试**

```swift
import XCTest
import Foundation
@testable import BaseAgentCore

/// 内置 reranker 的打包清单与默认配置解析。
final class BundledRerankerTests: XCTestCase {
    private static let modelName = "jina-reranker-v2-base-multilingual-onnx"

    private var temporaryRoot: URL!

    override func setUpWithError() throws {
        temporaryRoot = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("bundled-reranker-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: temporaryRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        unsetenv("BASE_AGENT_MODELS_DIR")
        LocalModelFactory.shared.clearCaches()
        if let temporaryRoot { try? FileManager.default.removeItem(at: temporaryRoot) }
    }

    /// 打包脚本必须复制 reranker 的全部运行时文件，避免发布包缺权重。
    func testPackageScriptBundlesRerankerRuntimeFiles() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let script = try String(
            contentsOf: repoRoot.appendingPathComponent("scripts/package-app.sh"),
            encoding: .utf8)

        XCTAssertTrue(
            script.contains("RERANKER_NAME=\"\(Self.modelName)\""),
            "scripts/package-app.sh 未声明内置 reranker 目录")
        for file in [
            "model_quantized.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "config.json",
        ] {
            XCTAssertTrue(
                script.contains(file),
                "scripts/package-app.sh 未复制 reranker 运行时文件：\(file)")
        }
    }

    /// 模拟 App bundle 的 Contents/Resources/models，默认配置应命中内置 reranker。
    func testDefaultConfigResolvesBundledReranker() throws {
        let (modelsRoot, directory) = try makeFakeBundledReranker()
        setenv("BASE_AGENT_MODELS_DIR", modelsRoot.path, 1)

        let config = AgentConfig()
        let resolved = try LocalModelRegistry.resolveDirectory(
            kind: .reranker,
            value: config.localRerankerModelPath,
            config: config)
        XCTAssertEqual(resolved.standardizedFileURL.path,
                       directory.standardizedFileURL.path)

        let provider = try XCTUnwrap(
            LocalModelFactory.shared.makeReranker(config: config))
        XCTAssertTrue(provider is ONNXReranker, "默认配置应命中内置 ONNX reranker：\(provider)")
        XCTAssertEqual(provider.identifier, "local:onnx:\(Self.modelName)")
    }

    /// 用户完全没有填写 reranker 路径/模型名时，应回落到内置默认模型。
    func testEmptyRerankerConfigUsesBundledDefault() throws {
        let (modelsRoot, directory) = try makeFakeBundledReranker()
        setenv("BASE_AGENT_MODELS_DIR", modelsRoot.path, 1)

        var config = AgentConfig()
        config.localRerankerModel = ""
        config.localRerankerModelPath = ""
        config.ragRerankModel = ""

        XCTAssertEqual(LocalModelRegistry.defaultName(for: .reranker), Self.modelName)
        let resolved = try LocalModelRegistry.resolveDirectory(
            kind: .reranker, value: "", config: config)
        XCTAssertEqual(resolved.standardizedFileURL.path,
                       directory.standardizedFileURL.path)

        let provider = try XCTUnwrap(
            LocalModelFactory.shared.makeReranker(config: config))
        XCTAssertTrue(provider is ONNXReranker, "空配置应回落到内置 reranker：\(provider)")
    }

    private func makeFakeBundledReranker() throws -> (modelsRoot: URL, directory: URL) {
        let modelsRoot = temporaryRoot.appendingPathComponent(
            "Contents/Resources/models", isDirectory: true)
        let directory = modelsRoot.appendingPathComponent(Self.modelName, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let files = [
            "model_quantized.onnx": "fake onnx",
            "tokenizer.json": "{}",
            "tokenizer_config.json": "{}",
            "special_tokens_map.json": "{}",
            "config.json": "{}",
        ]
        for (name, content) in files {
            try Data(content.utf8).write(to: directory.appendingPathComponent(name))
        }
        return (modelsRoot, directory)
    }
}
```

- [ ] **Step 2: 写真实模型 E2E 测试**

```swift
import XCTest
import Foundation
@testable import BaseAgentCore

/// 真实 jina reranker：验证内置模型可加载并正确排序。
final class JinaRerankerE2ETests: XCTestCase {
    private static let modelName = "jina-reranker-v2-base-multilingual-onnx"

    func testFactoryReranksRelevantDocumentFirst() async throws {
        let root = EmbeddingBenchmarkTests.resolveModelsRoot()
        let directory = root.appendingPathComponent(Self.modelName, isDirectory: true)
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("model_quantized.onnx").path),
            "模型缺失：\(directory.path)")

        var config = AgentConfig()
        config.localRerankerBackend = "onnx"
        config.localRerankerModel = Self.modelName
        config.localRerankerModelPath = directory.path

        let provider = try XCTUnwrap(
            LocalModelFactory.shared.makeReranker(config: config))
        defer { LocalModelFactory.shared.clearCaches() }

        let documents = [
            "苹果公司发布了新款智能手机，搭载 A19 芯片，夜景拍摄和充电能力明显提升。",
            "机器学习通过数据训练模型，让计算机自动改进任务表现。",
            "红烧肉需要小火慢炖，加入冰糖上色，最后大火收汁。",
        ]
        let scores = try await provider.rerank(
            query: "苹果新手机拍照和充电有什么升级？", documents: documents)
        XCTAssertEqual(scores.count, documents.count)
        for score in scores {
            XCTAssertTrue(score >= 0 && score <= 1, "分数应归一化到 [0,1]，实际 \(score)")
        }
        XCTAssertGreaterThan(scores[0], scores[1])
        XCTAssertGreaterThan(scores[0], scores[2])
        print("[jina-reranker-e2e] scores="
            + scores.map { String(format: "%.4f", $0) }.joined(separator: ", "))
    }
}
```

- [ ] **Step 3: 运行测试确认 RED**

Run: `swift test --filter BundledRerankerTests`
Expected: `testPackageScriptBundlesRerankerRuntimeFiles` FAIL（脚本尚未声明 RERANKER）；两个解析测试 PASS。

Run: `swift test --filter JinaRerankerE2ETests`
Expected: PASS（本地模型存在时），记录实际分数。

- [ ] **Step 4: 提交测试**

```bash
git add Tests/BaseAgentCoreTests/BundledRerankerTests.swift \
        Tests/BaseAgentCoreTests/JinaRerankerE2ETests.swift
git commit -m "test(reranker): cover bundled model packaging and defaults"
```

### Task 2: 打包 reranker 运行时文件

**Files:**
- Modify: `scripts/package-app.sh`

- [ ] **Step 1: 在 embedding 打包块后新增 reranker 块**

```bash
# 默认 Reranker：jina-reranker-v2-base-multilingual ONNX int8 随 App 分发
#（model_quantized.onnx 约 267MB）。与 embedding 一样只复制运行时必需文件。
RERANKER_NAME="jina-reranker-v2-base-multilingual-onnx"
RERANKER_SRC="$ROOT/models/$RERANKER_NAME"
RERANKER_DST="$APP/Contents/Resources/models/$RERANKER_NAME"
if [[ -d "$RERANKER_SRC" ]]; then
  mkdir -p "$RERANKER_DST"
  for file in model_quantized.onnx tokenizer.json tokenizer_config.json \
              special_tokens_map.json config.json; do
    if [[ -f "$RERANKER_SRC/$file" ]]; then
      cp "$RERANKER_SRC/$file" "$RERANKER_DST/"
    fi
  done
  echo "已打包默认 reranker：$RERANKER_NAME"
else
  echo "警告：未找到 $RERANKER_SRC，App 内不含默认 reranker" >&2
fi
```

- [ ] **Step 2: 运行测试确认 GREEN**

Run: `swift test --filter BundledRerankerTests`
Expected: 3 个测试全部 PASS。

- [ ] **Step 3: 提交**

```bash
git add scripts/package-app.sh
git commit -m "build: bundle jina reranker in app"
```

### Task 3: 设置界面提示

**Files:**
- Modify: `Sources/BaseAgentApp/SettingsView.swift:754-772`（`modelPathControls`）
- Modify: `Sources/BaseAgentApp/SettingsView.swift:798-810`（`rerankControls`）

- [ ] **Step 1: 路径输入框补充“留空使用内置模型”提示**

在 `modelPathControls` 的 `TextField("可选", text: path)` 后追加：

```swift
.help("留空时使用应用内置的默认模型；仅在需要自定义模型时填写路径")
```

- [ ] **Step 2: 重排序开关帮助文案说明内置 reranker**

```swift
.help("开启后先用混合检索取候选，再由 reranker 重新打分排序，"
    + "按下面的 TopN 返回最终结果；关闭则直接用 RRF 排序结果。"
    + "路径保持默认或留空时使用 App 内置 jina reranker")
```

- [ ] **Step 3: 构建验证**

Run: `swift build`
Expected: BUILD SUCCEEDED。

- [ ] **Step 4: 提交**

```bash
git add Sources/BaseAgentApp/SettingsView.swift
git commit -m "feat(settings): explain bundled reranker default"
```

### Task 4: 文档、全量回归与打包

**Files:**
- Modify: `README.md`
- Modify: `docs/软件设计文档.md`

- [ ] **Step 1: 更新 README**

- 默认模型说明增加：默认 Reranker 为随 app 打包的 `jina-reranker-v2-base-multilingual-onnx`，未配置时开箱可用。
- 打包/体积说明更新：App 约 640MB、DMG 约 480–520MB。
- 测试数量按实际全量结果更新。

- [ ] **Step 2: 更新设计文档**

- 5.4 检索章节说明默认重排使用内置 jina reranker。
- 6.3/6.3.1 标注 ONNX reranker 为内置默认模型。
- 测试章节加入 `BundledRerankerTests` / `JinaRerankerE2ETests`。

- [ ] **Step 3: 全量测试**

Run: `swift test --disable-sandbox`
Expected: 0 failures；记录实际 tests/skipped。

- [ ] **Step 4: Release 打包并验证内容**

Run: `swift build -c release --disable-sandbox && scripts/package-app.sh release`
Expected: 输出“已打包默认 reranker”；`du -sh dist/BaseAgent.app` 约 640MB；
`ls dist/BaseAgent.app/Contents/Resources/models/jina-reranker-v2-base-multilingual-onnx`
包含 5 个运行时文件。

- [ ] **Step 5: 生成并验证 DMG**

```bash
STAGE=$(mktemp -d /tmp/baseagent-dmg.XXXXXX)
cp -R dist/BaseAgent.app "$STAGE/"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "BaseAgent 1.2.4" -srcfolder "$STAGE" -ov -format UDZO \
  dist/BaseAgent-1.2.4.dmg
rm -rf "$STAGE"
hdiutil verify dist/BaseAgent-1.2.4.dmg
```

Expected: checksum VALID；挂载后能看到 App 与内置 reranker 文件。

- [ ] **Step 6: 提交文档**

```bash
git add README.md docs/软件设计文档.md
git commit -m "docs: document bundled reranker"
```

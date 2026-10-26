import XCTest
import Foundation
@testable import BaseAgentCore

/// 两个 reranker 方案的对比基准：
/// - A：`models/bge-reranker-v2-m3-mlx-mxfp8`（MLX 加载，24 层 XLM-R large，GPU）
/// - B：`models/bge-reranker-base-ONNX/model_quantized.onnx`（ONNX Runtime，12 层 XLM-R base，强制 CPU）
///
/// 需要真实模型与语料，默认跳过；设置 `RERANK_BENCH=1` 启用。
/// 可选环境变量：
/// - `RERANK_BENCH_MODELS_DIR`：模型根目录（默认仓库 models）
/// - `RERANK_BENCH_CORPUS`：语料目录（默认 SSD 工作区 rag_data/documents）
/// 报告输出到 `/tmp/rerank_bench_report.md` 并打印摘要。
///
/// ONNX 强制 CPU 的说明：项目 OrtSession 封装不追加任何 ExecutionProvider
/// （无 CoreML EP），仅设置 intraOpThreads=4，本就是纯 CPU 执行，符合要求。
final class RerankerBenchmarkTests: XCTestCase {

    private static let mlxModelName = "bge-reranker-v2-m3-mlx-mxfp8"
    private static let onnxModelName = "bge-reranker-base-ONNX"

    /// 与 EmbeddingBenchmarkTests 相同的 10 条中文 query 与目标段落（真实书籍原文）。
    private static var targets: [EmbeddingBenchmarkTests.Target] { EmbeddingBenchmarkTests.targets }

    /// 长块尾部检索：答案位于 1664 字长块的后半段，base 512 token 截断后不可见。
    private static let longTailQuery = "成年中期的发展任务是什么？"
    private static let longTailSignature = "7.繁殖感对停滞感"

    func testBenchmark() async throws {
        guard ProcessInfo.processInfo.environment["RERANK_BENCH"] == "1" else {
            throw XCTSkip("未设置 RERANK_BENCH=1，跳过真实模型基准测试")
        }

        let modelsRoot = EmbeddingBenchmarkTests.resolveModelsRoot()
        let corpusDir = ProcessInfo.processInfo.environment["RERANK_BENCH_CORPUS"]
            ?? "/Volumes/SSD/agent-space/rag_data/book-clean-34e533a6/documents"
        guard FileManager.default.fileExists(atPath: corpusDir) else {
            throw XCTSkip("语料目录不存在：\(corpusDir)")
        }

        let mlxDir = modelsRoot.appendingPathComponent(Self.mlxModelName, isDirectory: true)
        let onnxDir = modelsRoot.appendingPathComponent(Self.onnxModelName, isDirectory: true)
        guard FileManager.default.fileExists(
            atPath: mlxDir.appendingPathComponent("model.safetensors").path) else {
            throw XCTSkip("模型缺失：\(mlxDir.path)")
        }
        guard FileManager.default.fileExists(
            atPath: onnxDir.appendingPathComponent("model_quantized.onnx").path) else {
            throw XCTSkip("模型缺失：\(onnxDir.path)")
        }

        // ONNX 模型文件解析：目录下无 model.onnx，应命中用户指定的 model_quantized.onnx。
        let resolvedONNX = try LocalModelRegistry.resolveONNXModel(in: onnxDir, kind: .reranker)
        XCTAssertEqual(resolvedONNX.lastPathComponent, "model_quantized.onnx",
                       "ONNX 方案应解析到 model_quantized.onnx，实际：\(resolvedONNX.path)")

        // 从语料构建干扰块（确定性采样），与 embedding 基准同一语料。
        let corpus = try EmbeddingBenchmarkTests.buildCorpus(corpusDir: corpusDir)
        XCTAssertGreaterThanOrEqual(corpus.count, 60, "干扰块过少")
        print("[bench] 语料干扰块：\(corpus.count)")

        EmbeddingBenchmarkTests.configureMetallib()

        var report = "# Reranker 对比基准：bge-reranker-v2-m3-mlx-mxfp8 (MLX/GPU) " +
            "vs bge-reranker-base-ONNX model_quantized (CPU only)\n\n"
        report += "- 方案A：bge-reranker-v2-m3（XLM-R large 24 层，mxfp8，MLX 加载走 GPU）\n"
        report += "- 方案B：bge-reranker-base（XLM-R base 12 层，ONNX int8，" +
            "OrtSession 无 CoreML EP 纯 CPU，intraOpThreads=4）\n\n"

        // ---------- tokenizer 统计（上下文/截断影响） ----------
        report += "## Tokenizer 统计（语料 \(corpus.count) 块 + 10 目标块）\n\n"
        report += await Self.tokenizerStats(directory: mlxDir, corpus: corpus, label: "v2-m3-mlx（8192 上下文）")
        report += await Self.tokenizerStats(directory: onnxDir, corpus: corpus, label: "base-onnx（512 上下文）")

        // ---------- Provider 构建 ----------
        let mlx = MLXReranker(directory: mlxDir, modelName: Self.mlxModelName)
        let onnx = ONNXReranker(directory: onnxDir, modelName: Self.onnxModelName)
        defer { Task { await mlx.unload(); await onnx.unload() } }

        // ---------- 加载与速度 ----------
        let sample = Array(corpus.prefix(32))
        let mlxLoad = try await Self.measureLoad(provider: mlx, documents: sample)
        let onnxLoad = try await Self.measureLoad(provider: onnx, documents: sample)
        report += "\n## 加载与速度（query 固定为「\(Self.targets[0].query)」）\n\n"
        report += "| 指标 | v2-m3-mlx-mxfp8 (GPU) | base-onnx-int8 (CPU) |\n|---|---|---|\n"
        report += "| 加载+首次 rerank 32块 (ms) | \(mlxLoad) | \(onnxLoad) |\n"

        let mlxSmall = try await Self.measureBatchLatency(provider: mlx, documents: Array(sample.prefix(8)))
        let onnxSmall = try await Self.measureBatchLatency(provider: onnx, documents: Array(sample.prefix(8)))
        report += "| 8 短块批量延迟 (ms) | \(mlxSmall) | \(onnxSmall) |\n"

        let mlxThroughput = try await Self.measureThroughput(provider: mlx, documents: sample)
        let onnxThroughput = try await Self.measureThroughput(provider: onnx, documents: sample)
        report += "| 批量吞吐 32块/批 (pairs/s) | \(mlxThroughput) | \(onnxThroughput) |\n"

        // 长文档（>512 字）批延迟：v2-m3 不截断，base 截断到 512 token。
        let longBatchDocs = Array(corpus.filter { $0.count > 512 }.prefix(8))
        if longBatchDocs.count >= 4 {
            let mlxLong = try await Self.measureBatchLatency(provider: mlx, documents: longBatchDocs)
            let onnxLong = try await Self.measureBatchLatency(provider: onnx, documents: longBatchDocs)
            report += "| 8 长块(>512字)批量延迟 (ms) | \(mlxLong) | \(onnxLong) |\n"
        }

        // ---------- 排序质量 ----------
        let docTexts = corpus + Self.targets.map(\.passage)
        let targetBase = corpus.count

        report += "\n## 排序质量（10 条中文 query，一次 rerank \(docTexts.count) 候选）\n\n"
        report += "| # | Query | mlx top1分 | onnx top1分 | mlx命中 | onnx命中 |\n|---|---|---|---|---|---|\n"
        var mlxHits1 = 0, mlxHits3 = 0, onnxHits1 = 0, onnxHits3 = 0
        var mlxRR = 0.0, onnxRR = 0.0
        var mlxGaps: [Double] = [], onnxGaps: [Double] = []
        var mlxTargetScores: [Double] = [], onnxTargetScores: [Double] = []
        for (index, target) in Self.targets.enumerated() {
            let targetIndex = targetBase + index
            let mlxRank = try await Self.rank(
                provider: mlx, query: target.query, documents: docTexts, targetIndex: targetIndex,
                signature: target.signature)
            let onnxRank = try await Self.rank(
                provider: onnx, query: target.query, documents: docTexts, targetIndex: targetIndex,
                signature: target.signature)
            if mlxRank.position == 1 { mlxHits1 += 1 }
            if mlxRank.position <= 3 { mlxHits3 += 1 }
            mlxRR += mlxRank.rr
            mlxGaps.append(mlxRank.targetScore - mlxRank.medianUnrelated)
            mlxTargetScores.append(mlxRank.targetScore)
            if onnxRank.position == 1 { onnxHits1 += 1 }
            if onnxRank.position <= 3 { onnxHits3 += 1 }
            onnxRR += onnxRank.rr
            onnxGaps.append(onnxRank.targetScore - onnxRank.medianUnrelated)
            onnxTargetScores.append(onnxRank.targetScore)
            let mark: (Int) -> String = { $0 == 1 ? "✅1" : ($0 <= 3 ? "🟡\($0)" : "❌\($0)") }
            report += "| \(index + 1) | \(target.query) " +
                "| \(String(format: "%.3f", mlxRank.targetScore)) " +
                "| \(String(format: "%.3f", onnxRank.targetScore)) " +
                "| \(mark(mlxRank.position)) | \(mark(onnxRank.position)) |\n"
        }
        let total = Double(Self.targets.count)
        let mlxMedianGap = Self.median(mlxGaps)
        let onnxMedianGap = Self.median(onnxGaps)
        report += "\n| 指标 | v2-m3-mlx-mxfp8 | base-onnx-int8 |\n|---|---|---|\n"
        report += "| Recall@1 | \(mlxHits1)/\(Self.targets.count) | \(onnxHits1)/\(Self.targets.count) |\n"
        report += "| Recall@3 | \(mlxHits3)/\(Self.targets.count) | \(onnxHits3)/\(Self.targets.count) |\n"
        report += "| MRR | \(String(format: "%.3f", mlxRR / total)) | \(String(format: "%.3f", onnxRR / total)) |\n"
        report += "| 中位区分度（目标分-无关中位分） | \(String(format: "%.3f", mlxMedianGap)) " +
            "| \(String(format: "%.3f", onnxMedianGap)) |\n"

        // ---------- 阈值视角（项目现有 0.7 重排阈值） ----------
        let mlxPass = mlxTargetScores.filter { $0 >= 0.7 }.count
        let onnxPass = onnxTargetScores.filter { $0 >= 0.7 }.count
        report += "\n## 0.7 阈值视角（目标块分数 ≥ 0.7 的比例）\n\n"
        report += "| 方案 | 目标块分数 | 通过 0.7 |\n|---|---|---|\n"
        report += "| v2-m3-mlx | \(Self.formatScores(mlxTargetScores)) | \(mlxPass)/10 |\n"
        report += "| base-onnx | \(Self.formatScores(onnxTargetScores)) | \(onnxPass)/10 |\n"

        // ---------- 长块尾部检索（截断影响） ----------
        let longTailDocs = [Self.targets[0].passage] + Array(corpus.prefix(8))
        let mlxLong = try await Self.rank(
            provider: mlx, query: Self.longTailQuery, documents: longTailDocs, targetIndex: 0,
            signature: Self.longTailSignature)
        let onnxLong = try await Self.rank(
            provider: onnx, query: Self.longTailQuery, documents: longTailDocs, targetIndex: 0,
            signature: Self.longTailSignature)
        report += "\n## 长块尾部检索（\(Self.targets[0].passage.count) 字块，答案在后半段）\n\n"
        func cell(_ rank: RankResult) -> String {
            rank.position == Int.max
                ? "未命中 | - | \(String(format: "%.3f", rank.targetScore))"
                : "命中 | \(rank.position) | \(String(format: "%.3f", rank.targetScore))"
        }
        report += "| Query：\(Self.longTailQuery) | 命中 | 排名 | 目标分 |\n|---|---|---|---|\n"
        report += "| v2-m3-mlx（8192 不截断） | \(cell(mlxLong)) |\n"
        report += "| base-onnx（512 截断） | \(cell(onnxLong)) |\n"

        try report.write(toFile: "/tmp/rerank_bench_report.md", atomically: true, encoding: .utf8)
        print("[bench] 报告已写入 /tmp/rerank_bench_report.md")
        print(report)
    }

    // MARK: - 工具

    private struct RankResult {
        /// 1-based；未命中为 Int.max。
        let position: Int
        let targetScore: Double
        let medianUnrelated: Double
        var rr: Double { position == Int.max ? 0 : 1.0 / Double(position) }
    }

    private static func rank(
        provider: RerankProvider,
        query: String,
        documents: [String],
        targetIndex: Int,
        signature: String
    ) async throws -> RankResult {
        let scores = try await provider.rerank(query: query, documents: documents)
        XCTAssertEqual(scores.count, documents.count)
        let scored = scores.enumerated().map { (index: $0.offset, score: $0.element) }
            .sorted { $0.score > $1.score }
        let needle = EmbeddingBenchmarkTests.normalize(signature)
        let position = scored.firstIndex {
            EmbeddingBenchmarkTests.normalize(documents[$0.index]).contains(needle)
        }.map { $0 + 1 } ?? Int.max
        let unrelated = scores.enumerated()
            .filter { $0.offset != targetIndex }
            .map(\.element)
            .sorted()
        let median = unrelated.isEmpty ? 0 : unrelated[unrelated.count / 2]
        return RankResult(position: position,
                          targetScore: scores[targetIndex],
                          medianUnrelated: median)
    }

    private static func tokenizerStats(directory: URL, corpus: [String], label: String) async -> String {
        guard let tokenizer = try? await LocalTokenizer.load(directory: directory) else {
            return "\(label)：tokenizer 加载失败\n"
        }
        var tokens = 0, chars = 0, over512 = 0, total = 0
        for text in Array(corpus.prefix(60)) + targets.map(\.passage) {
            let ids = tokenizer.encode(text, addSpecialTokens: true)
            tokens += ids.count
            chars += text.count
            total += 1
            if ids.count > 512 { over512 += 1 }
        }
        guard chars > 0 else { return "" }
        return "\(label)：均值 \(String(format: "%.2f", Double(tokens) / Double(chars))) token/字，" +
            "超 512 token 块占比 \(String(format: "%.0f%%", Double(over512) / Double(total) * 100))\n"
    }

    private static func measureLoad(provider: RerankProvider, documents: [String]) async throws -> Int {
        let start = Date()
        _ = try await provider.rerank(query: targets[0].query, documents: documents)
        return Int(Date().timeIntervalSince(start) * 1000)
    }

    private static func measureBatchLatency(
        provider: RerankProvider, documents: [String]
    ) async throws -> String {
        _ = try await provider.rerank(query: targets[0].query, documents: documents) // 预热
        let runs = 5
        let start = Date()
        for _ in 0..<runs {
            _ = try await provider.rerank(query: targets[0].query, documents: documents)
        }
        let ms = Date().timeIntervalSince(start) * 1000 / Double(runs)
        return String(format: "%.0f", ms)
    }

    private static func measureThroughput(
        provider: RerankProvider, documents: [String]
    ) async throws -> String {
        _ = try await provider.rerank(query: targets[0].query, documents: documents) // 预热
        let runs = 3
        let start = Date()
        var count = 0
        for _ in 0..<runs {
            _ = try await provider.rerank(query: targets[0].query, documents: documents)
            count += documents.count
        }
        let seconds = Date().timeIntervalSince(start)
        return String(format: "%.1f", Double(count) / seconds)
    }

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        return sorted[sorted.count / 2]
    }

    private static func formatScores(_ values: [Double]) -> String {
        values.map { String(format: "%.2f", $0) }.joined(separator: ", ")
    }
}

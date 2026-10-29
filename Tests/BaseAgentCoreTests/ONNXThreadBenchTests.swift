import XCTest
@testable import BaseAgentCore

/// ONNX embedding 的「并发 run × intra-op 线程」实测。
///
/// 设置 `ONNX_THREAD_BENCH=1` 启用；默认跳过，避免普通测试跑真实模型。
/// 对比 1×8 / 2×4 / 2×8：同一批文本、同样两路并发调用，看墙钟吞吐。
final class ONNXThreadBenchTests: XCTestCase {
    func testConcurrencyAndThreadConfiguration() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["ONNX_THREAD_BENCH"] == "1",
            "未设置 ONNX_THREAD_BENCH=1")

        let modelName = "jina-embeddings-v3-onnx-int4"
        let directory = EmbeddingBenchmarkTests.resolveModelsRoot()
            .appendingPathComponent(modelName)
        guard FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("model.onnx").path) else {
            throw XCTSkip("ONNX 模型缺失：\(directory.path)")
        }

        let textCount = Int(ProcessInfo.processInfo.environment["ONNX_THREAD_BENCH_TEXTS"] ?? "") ?? 24
        let targetTokens = Int(ProcessInfo.processInfo.environment["ONNX_THREAD_BENCH_TOKENS"] ?? "") ?? 300
        let configs: [(runs: Int, threads: Int)] = [(1, 8), (2, 4), (2, 8)]

        // 用真实 tokenizer 把样本压到目标长度，保证不同配置处理同样多 token。
        let tokenizerProvider = ONNXEmbeddingProvider(directory: directory, modelName: modelName)
        let counterValue = await tokenizerProvider.makeTokenCounter()
        let counter = try XCTUnwrap(counterValue)
        let texts = Self.makeTexts(count: textCount, targetTokens: targetTokens, counter: counter)
        let tokens = texts.reduce(0) { $0 + counter($1) }
        print("[onnx-thread-bench] texts=\(texts.count) totalTokens=\(tokens) "
            + "tokens/text≈\(tokens / max(texts.count, 1)) cores=\(ProcessInfo.processInfo.activeProcessorCount)")

        let half = texts.count / 2
        for round in 0..<2 {
            // 第二轮倒序，抵消温度/频率漂移。
            let order = round == 0 ? Array(configs.indices) : Array(configs.indices.reversed())
            for index in order {
                let config = configs[index]
                let provider = ONNXEmbeddingProvider(
                    directory: directory, modelName: modelName,
                    concurrentRuns: config.runs, intraOpThreads: config.threads)
                let scheduler = AdaptiveEmbeddingScheduler(
                    base: provider,
                    options: .init(minBatchSize: half, maxBatchSize: half,
                                   initialBatchSize: half, step: 1,
                                   memoryLimitMB: 32_768, throughputTolerance: 0.5,
                                   probeBatches: 16, probeSeconds: 30,
                                   tokenBudgetMin: 8_192, tokenBudgetMax: 8_192,
                                   initialTokenBudget: 8_192, tokenBudgetStep: 128),
                    log: { _ in })
                // 预热 session / 线程池，不计入吞吐。
                _ = try await provider.embedDocuments(["warmup"])

                let started = Date()
                async let first = scheduler.embedDocuments(Array(texts[0..<half]))
                async let second = scheduler.embedDocuments(Array(texts[half...]))
                let (firstVectors, secondVectors) = try await (first, second)
                let seconds = Date().timeIntervalSince(started)
                XCTAssertEqual(firstVectors.count + secondVectors.count, texts.count)

                let metrics = scheduler.metricsSnapshot()
                print(String(format: "[onnx-thread-bench] r%d %d×%d: %.2fs  %.0f tokens/s  peak %.0fMB",
                             round + 1, config.runs, config.threads, seconds,
                             Double(tokens) / max(seconds, 0.001), metrics.peakMemoryMB))
                await provider.unload()
            }
        }
        await tokenizerProvider.unload()
    }

    /// 构造接近真实切片长度的中文样本文本。
    private static func makeTexts(count: Int, targetTokens: Int,
                                  counter: (String) -> Int) -> [String] {
        let paragraph = """
        检索链路关注切片质量、向量维度、量化误差与召回排序，工程团队还要权衡吞吐、\
        内存占用与恢复时间。数据同步需要处理增量更新与冲突合并，监控体系覆盖指标、\
        日志与链路三个维度，容量评估依据历史峰值与增长趋势建模，并预留安全水位。
        """
        let base = String(repeating: paragraph, count: 20)
        var characters = min(base.count, targetTokens * 2)
        var sample = String(base.prefix(characters))
        while characters > 40, counter(sample) > targetTokens {
            characters -= 20
            sample = String(base.prefix(characters))
        }
        return (0..<count).map { index in
            "文档\(index + 1)：\(sample)"
        }
    }
}

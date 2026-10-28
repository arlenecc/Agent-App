import XCTest
import Foundation
import MLX
@testable import BaseAgentCore

/// 真实 jina-embeddings-v3 MLX 模型上对比 MLX buffer cache 512MB / 1GB / 2GB
/// 对 embedding 吞吐的影响。
///
/// 默认跳过；设置 `EMBED_CACHE_BENCH=1` 启用：
///
/// ```bash
/// EMBED_CACHE_BENCH=1 swift test --disable-sandbox \
///   --filter JinaV3CacheBenchmarkTests
/// ```
///
/// 可选环境变量：
/// - `EMBED_CACHE_BATCH`：每批文本数，默认 32（与调度器默认上限一致）
/// - `EMBED_CACHE_TEXTS`：参与计时的文本数，默认 32
/// - `EMBED_CACHE_TOKENS`：每条文本目标 token 数，默认 780（贴近 800 切片）
final class JinaV3CacheBenchmarkTests: XCTestCase {
    private static let megabyte = 1_024 * 1_024

    func testCacheLimitImpactOnEmbeddingThroughput() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["EMBED_CACHE_BENCH"] == "1",
            "未设置 EMBED_CACHE_BENCH=1，跳过 MLX cache 基准")

        let environment = ProcessInfo.processInfo.environment
        let batchSize = Int(environment["EMBED_CACHE_BATCH"] ?? "") ?? 32
        let textCount = Int(environment["EMBED_CACHE_TEXTS"] ?? "") ?? 32
        let targetTokens = Int(environment["EMBED_CACHE_TOKENS"] ?? "") ?? 780
        let rounds = Int(environment["EMBED_CACHE_ROUNDS"] ?? "") ?? 3
        let cooldown = Double(environment["EMBED_CACHE_COOLDOWN"] ?? "") ?? 0
        let limits = (environment["EMBED_CACHE_LIMITS"] ?? "512,1024,2048,2048,1024,512")
            .split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }

        let modelsRoot = EmbeddingBenchmarkTests.resolveModelsRoot()
        let directory = modelsRoot.appendingPathComponent("jina-embeddings-v3-mlx-int4")
        guard FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("weights.safetensors").path)
            || FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("model.safetensors").path) else {
            throw XCTSkip("jina 模型缺失：\(directory.path)")
        }
        EmbeddingBenchmarkTests.configureMetallib()

        let provider = JinaV3MLXEmbeddingProvider(
            directory: directory, modelName: "jina-embeddings-v3-mlx-int4")
        defer { Task { await provider.unload() } }
        let counterValue = await provider.makeTokenCounter()
        let counter = try XCTUnwrap(counterValue, "应能用 tokenizer.json 构建计数器")
        let texts = Self.makeTexts(counter: counter, count: textCount,
                                   targetTokens: targetTokens)
        XCTAssertEqual(texts.count, textCount)
        let tokensPerText = Double(texts.map(counter).reduce(0, +)) / Double(textCount)

        // 预编译所有形状的 Metal kernel：这一步不计时，避免首个配置吃亏。
        MLX.Memory.cacheLimit = 2_048 * Self.megabyte
        MLX.Memory.clearCache()
        _ = try await provider.embedDocuments(Array(texts.prefix(min(batchSize, textCount))))

        print("[cache-bench] 物理内存 \(ProcessInfo.processInfo.physicalMemory / 1_048_576)MB；"
            + "batch=\(batchSize) texts=\(textCount) "
            + "tokens/text≈\(String(format: "%.0f", tokensPerText))")
        print("| cache MB | 批次 | 中位耗时 s | texts/s | tokens/s | cache MB(after) "
            + "| peak MB | footprint MB |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|")

        // 默认交错重复两轮，抵消温度/频率漂移：512 → 1024 → 2048 → 2048 → 1024 → 512。
        var results: [Int: [Double]] = [:]
        for (index, limit) in limits.enumerated() {
            MLX.Memory.cacheLimit = limit * Self.megabyte
            MLX.Memory.clearCache()
            // 用同形状 batch 预热，让该上限下的 cache 填满。
            _ = try await provider.embedDocuments(Array(texts.prefix(min(batchSize, textCount))))

            var times: [Double] = []
            var peakMB = 0.0
            var cacheMB = 0.0
            let footprintBefore = ProcessMemory.footprintMB()
            MLX.Memory.peakMemory = 0
            for _ in 0 ..< rounds {
                let batch = Array(texts.prefix(min(batchSize, textCount)))
                let start = Date()
                let vectors = try await provider.embedDocuments(batch)
                let seconds = Date().timeIntervalSince(start)
                XCTAssertEqual(vectors.count, batch.count)
                XCTAssertEqual(vectors.first?.count, 1_024)
                times.append(seconds)
                peakMB = max(peakMB, Double(MLX.Memory.peakMemory) / 1_048_576)
                cacheMB = Double(MLX.Memory.cacheMemory) / 1_048_576
            }
            let median = times.sorted()[times.count / 2]
            let footprint = ProcessMemory.footprintMB() - footprintBefore
            let measured = Double(min(batchSize, textCount))
            let textsPerSecond = measured / max(median, 0.0001)
            let tokensPerSecond = textsPerSecond * tokensPerText
            results[limit, default: []].append(median)
            print(String(format: "| %d | %d | %.3f | %.2f | %.0f | %.0f | %.0f | %.0f |",
                         limit, Int(measured), median, textsPerSecond, tokensPerSecond,
                         cacheMB, peakMB, footprint))
            if cooldown > 0, index < limits.count - 1 {
                Thread.sleep(forTimeInterval: cooldown)
            }
        }

        for limit in Set(limits).sorted() {
            let samples = try XCTUnwrap(results[limit]).sorted()
            let median = samples[samples.count / 2]
            print("[cache-bench] cache \(limit)MB 中位 "
                + String(format: "%.3fs 最快 %.3fs 样本 %@", median, samples[0],
                         samples.map { String(format: "%.3f", $0) }.joined(separator: "/")))
        }
    }

    /// 用 tokenizer 校准出约 `targetTokens` 的中文文本，并给每条加不同编号，
    /// 保持长度一致以便 batch 形状稳定。
    private static func makeTexts(counter: (String) -> Int, count: Int,
                                  targetTokens: Int) -> [String] {
        let paragraph = """
        云计算平台通过容器编排实现弹性伸缩，监控体系覆盖指标、日志与链路三个维度，\
        容量评估依据历史峰值与增长趋势建模，并预留合理的安全水位。\
        数据同步需要处理增量更新与冲突合并，检索链路则关注切片质量、向量维度、\
        量化误差与召回排序。工程团队还要权衡吞吐、内存占用与恢复时间，\
        在发布流程中引入灰度、回滚和可观测性，确保长稳运行。
        """
        let base = String(repeating: paragraph, count: 40)
        var characters = min(base.count, targetTokens * 2)
        var sample = String(base.prefix(characters))
        while characters > 40, counter(sample) > targetTokens {
            characters -= 20
            sample = String(base.prefix(characters))
        }
        while characters < base.count, counter(sample) < targetTokens - 10 {
            characters += 10
            sample = String(base.prefix(characters))
        }
        return (0 ..< count).map { index in
            "文档\(index + 1)：\(sample)"
        }
    }
}

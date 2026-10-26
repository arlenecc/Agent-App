import XCTest
import Foundation
import MLX
@testable import BaseAgentCore

/// 子批自适应参数模拟：在真实 bge-small-zh-v1.5-mlx 上测量
/// 「batch size → 吞吐 / 进程内存」曲线，并用真实调度器跑一遍上探过程。
/// 需要真实模型，默认跳过；设置 `EMBED_ADAPT_BENCH=1` 启用。
///
/// 输出：`/tmp/embed_adaptive_bench_report.md`
final class EmbeddingAdaptiveBenchmarkTests: XCTestCase {

    private static let batchSizes = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256]

    func testBatchScalingAndAdaptiveController() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["EMBED_ADAPT_BENCH"] == "1" else {
            throw XCTSkip("未设置 EMBED_ADAPT_BENCH=1，跳过子批自适应基准")
        }
        let targetTokens = Int(environment["EMBED_ADAPT_TOKENS"] ?? "") ?? 480
        let textCount = Int(environment["EMBED_ADAPT_TEXTS"] ?? "") ?? 4_096
        let scaleOnly = environment["EMBED_ADAPT_SCALE_ONLY"] == "1"
        let defaults = AgentConfig()
        let maxBatch = defaults.ragEmbeddingMaxBatch
        let memoryLimitMB = defaults.ragEmbeddingMemoryLimitMB
        let tolerance = Double(environment["EMBED_ADAPT_TOLERANCE"] ?? "")
            ?? defaults.ragEmbeddingThroughputTolerance
        let directory = EmbeddingBenchmarkTests.resolveModelsRoot()
            .appendingPathComponent("bge-small-zh-v1.5-mlx")
        guard FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("model.safetensors").path) else {
            throw XCTSkip("bge 模型缺失：\(directory.path)")
        }
        EmbeddingBenchmarkTests.configureMetallib()
        // 与生产默认一致：cache 1GB、全局上限 min(物理内存 65%, 9.6GB)。
        MLX.Memory.cacheLimit = 1_024 * 1_024 * 1_024
        MLX.Memory.memoryLimit = 9_830 * 1_024 * 1_024

        let provider = MLXEmbeddingProvider(directory: directory,
                                            modelName: "bge-small-zh-v1.5-mlx")
        defer { Task { await provider.unload() } }
        let counterValue = await provider.makeTokenCounter()
        let counter = try XCTUnwrap(counterValue, "应能用 tokenizer.json 构建计数器")
        let texts = Self.makeTexts(counter: counter, count: textCount,
                                   targetTokens: targetTokens)
        let physicalMB = Int(ProcessInfo.processInfo.physicalMemory / 1_024 / 1_024)

        print("[adapt-bench] 物理内存 \(physicalMB)MB；目标 \(targetTokens) token；"
            + "首条样本 \(counter(texts[0])) token / \(texts[0].count) 字")

        // 预热：加载权重并填满首轮 cache。
        _ = try await provider.embedDocuments(Array(texts.prefix(16)))

        var report = "# Embedding 子批自适应基准\n\n"
        report += "- 机器：\(physicalMB)MB 物理内存\n"
        report += "- 模型：bge-small-zh-v1.5-mlx（512 窗口）\n"
        report += "- 样本：\(texts.count) 条，目标 \(targetTokens) token，"
        report += "首条约 \(counter(texts[0])) token\n"
        report += "- MLX：cacheLimit 1024MB，memoryLimit min(65%, 9.6GB)\n\n"

        // ---------- 1) batch size → 吞吐/内存 ----------
        report += "## 1) batch size 扫描（每档 3 轮取中位）\n\n"
        report += "| batch | 中位耗时 ms | 吞吐 片/秒 | 每片 ms | footprint 前→后 MB | MLX peak MB | active/cache MB |\n"
        report += "|---:|---:|---:|---:|---:|---:|---:|\n"
        var throughputBySize: [Int: Double] = [:]
        for size in Self.batchSizes {
            let batch = Array(texts.prefix(size))
            MLX.Memory.peakMemory = 0
            let before = ProcessMemory.footprintMB()
            var times: [Double] = []
            for _ in 0..<3 {
                let start = Date()
                _ = try await provider.embedDocuments(batch)
                times.append(Date().timeIntervalSince(start))
            }
            let median = times.sorted()[times.count / 2]
            let after = ProcessMemory.footprintMB()
            let peak = Double(MLX.Memory.peakMemory) / 1_048_576
            let active = Double(MLX.Memory.activeMemory) / 1_048_576
            let cache = Double(MLX.Memory.cacheMemory) / 1_048_576
            let throughput = Double(size) / max(median, 0.0001)
            throughputBySize[size] = throughput
            report += String(format: "| %d | %.1f | %.1f | %.1f | %.0f → %.0f | %.0f | %.0f / %.0f |\n",
                             size, median * 1_000, throughput, median * 1_000 / Double(size),
                             before, after, peak, active, cache)
        }

        if scaleOnly {
            let reportPath = "/tmp/embed_adaptive_scale_\(counter(texts[0]))t.md"
            try report.write(toFile: reportPath, atomically: true, encoding: .utf8)
            print("[adapt-bench] 仅扫描模式，报告：\(reportPath)")
            print(report)
            return
        }

        // ---------- 2) 真实调度器上探过程 ----------
        report += "\n## 2) 真实 AdaptiveEmbeddingScheduler（16 / +2 / \(maxBatch)"
        report += " / \(memoryLimitMB)MB / ±\(Int((tolerance * 100).rounded()))%"
        report += " / \(defaults.ragEmbeddingProbeBatches) 批）\n\n"
        let recorder = RecordingEmbedding(base: provider)
        var logs: [String] = []
        let logLock = NSLock()
        let scheduler = AdaptiveEmbeddingScheduler(
            base: recorder,
            options: .init(minBatchSize: defaults.ragEmbeddingMinBatch,
                           maxBatchSize: maxBatch,
                           initialBatchSize: defaults.ragEmbeddingInitialBatch,
                           step: defaults.ragEmbeddingBatchStep,
                           memoryLimitMB: Double(memoryLimitMB),
                           throughputTolerance: tolerance,
                           probeBatches: defaults.ragEmbeddingProbeBatches,
                           probeSeconds: defaults.ragEmbeddingProbeSeconds),
            log: { line in
                print(line)
                logLock.lock()
                logs.append(line)
                logLock.unlock()
            })
        let start = Date()
        let vectors = try await scheduler.embedDocuments(texts)
        let totalSeconds = Date().timeIntervalSince(start)
        XCTAssertEqual(vectors.count, texts.count)
        let events = recorder.snapshot()
        let maxSize = events.map(\.size).max() ?? 0
        let maxFootprint = events.map(\.afterMB).max() ?? 0
        let probeHits = events.filter { $0.beforeMB > Double(memoryLimitMB) }.count
        let firstMaxIndex = events.firstIndex { $0.size == maxBatch }
        let chunksToMax = firstMaxIndex.map { index in
            events[0...index].reduce(0) { $0 + $1.size }
        }
        let secondsToMax = firstMaxIndex.map { index in
            events[0...index].reduce(0) { $0 + $1.seconds }
        }
        report += "| 指标 | 值 |\n|---|---|\n"
        report += String(format: "| 总耗时 | %.1f s |\n", totalSeconds)
        report += String(format: "| 总吞吐 | %.1f 片/秒 |\n",
                         Double(texts.count) / totalSeconds)
        report += "| 最终子批 | \(maxSize) |\n"
        if let chunksToMax, let secondsToMax {
            report += String(format: "| 到 %d 的耗时 | %.1f s |\n", maxBatch, secondsToMax)
            report += "| 到 \(maxBatch) 已处理 | \(chunksToMax) / \(texts.count) 片 |\n"
        } else {
            report += "| 到 \(maxBatch) | 未达到 |\n"
        }
        report += String(format: "| 内存峰值 | %.0f MB |\n", maxFootprint)
        report += "| \(memoryLimitMB)MB 阈值触发次数 | \(probeHits) |\n"
        let sizes = events.map(\.size)
        report += "| 子批轨迹（前 40 次） | \(sizes.prefix(40).map(String.init).joined(separator: "→")) |\n"
        report += "| 子批轨迹（后 20 次） | \(sizes.suffix(20).map(String.init).joined(separator: "→")) |\n"
        if !logs.isEmpty {
            report += "\n调参日志：\n\n```\n"
            report += logs.prefix(40).joined(separator: "\n")
            report += "\n```\n"
        }

        // ---------- 3) 用实测吞吐估算不同参数的爬坡成本 ----------
        report += "\n## 3) 爬坡成本估算（按实测吞吐插值）\n\n"
        report += "| 初始 | 步进 | 上限 | 到上限处理片数 | 到上限耗时 s |\n"
        report += "|---:|---:|---:|---:|---:|\n"
        let configs: [(Int, Int, Int)] = [
            (16, 2, 32),   // 当前默认
            (16, 2, 64),   // 旧默认
            (16, 4, 32),
            (16, 4, 64),
            (32, 2, 64),
            (32, 4, 64),
            (16, 8, 64),
            (16, 2, 128),
            (32, 8, 128),
            (16, 2, 256),
            (32, 8, 256),
        ]
        for (initial, step, maxSize) in configs {
            let (chunks, seconds) = Self.simulateRamp(
                initial: initial, step: step, maxSize: maxSize,
                probeBatches: defaults.ragEmbeddingProbeBatches,
                throughputBySize: throughputBySize)
            report += String(format: "| %d | %d | %d | %d | %.1f |\n",
                             initial, step, maxSize, chunks, seconds)
        }

        let reportPath = "/tmp/embed_adaptive_bench_report.md"
        try report.write(toFile: reportPath, atomically: true, encoding: .utf8)
        print("[adapt-bench] 报告：\(reportPath)")
        print(report)
    }

    // MARK: - 工具

    /// 生成固定 token 长度的中文样本；前缀编号让每条内容略有差异。
    private static func makeTexts(counter: (String) -> Int,
                                  count: Int,
                                  targetTokens: Int) -> [String] {
        let units = [
            "云计算技术近年来在数据存储、弹性伸缩和分布式调度方面发展迅速。",
            "企业纷纷将业务迁移上云，以降低运维成本并提升系统可用性。",
            "检索增强生成把向量检索与大语言模型结合起来，先找资料再回答。",
            "向量数据库通过近似最近邻算法，在百万级语料中快速召回相关内容。",
            "切片质量直接影响检索效果，过长会被模型截断，过短会丢失上下文。",
            "嵌入模型把文本映射成稠密向量，语义相近的句子在空间中距离更近。",
            "批处理能够提高 GPU 利用率，但批次过大会增加显存峰值与单批延迟。",
            "内存占用包括模型权重、中间激活值和可复用的缓冲池。",
        ]
        var base = ""
        var index = 0
        while counter(base + units[index % units.count]) <= targetTokens {
            base += units[index % units.count]
            index += 1
        }
        return (0..<count).map { "【\($0)】" + base }
    }

    /// 按 batch 吞吐插值估算：每档观察 2 批，到达上限即停。
    private static func simulateRamp(initial: Int,
                                     step: Int,
                                     maxSize: Int,
                                     probeBatches: Int,
                                     throughputBySize: [Int: Double]) -> (Int, Double) {
        var size = initial
        var chunks = 0
        var seconds = 0.0
        var guardCount = 0
        while size < maxSize, guardCount < 10_000 {
            guardCount += 1
            let throughput = interpolatedThroughput(size, table: throughputBySize)
            let batchSeconds = Double(size) / max(throughput, 0.0001)
            chunks += size * probeBatches
            seconds += batchSeconds * Double(probeBatches)
            size = min(maxSize, size + step)
        }
        return (chunks, seconds)
    }

    private static func interpolatedThroughput(_ size: Int,
                                               table: [Int: Double]) -> Double {
        let keys = table.keys.sorted()
        guard let first = keys.first, let last = keys.last else { return 1 }
        if size <= first { return table[first] ?? 1 }
        if size >= last { return table[last] ?? 1 }
        for (lower, upper) in zip(keys, keys.dropFirst()) where size <= upper {
            let lowThroughput = table[lower] ?? 1
            let highThroughput = table[upper] ?? lowThroughput
            let ratio = Double(size - lower) / Double(upper - lower)
            return lowThroughput + (highThroughput - lowThroughput) * ratio
        }
        return table[last] ?? 1
    }

    /// 记录每次底层调用的大小、耗时与进程 footprint。
    private final class RecordingEmbedding: EmbeddingProvider, @unchecked Sendable {
        struct Event {
            let size: Int
            let seconds: Double
            let beforeMB: Double
            let afterMB: Double
        }

        private let base: MLXEmbeddingProvider
        private let lock = NSLock()
        private var events: [Event] = []

        init(base: MLXEmbeddingProvider) {
            self.base = base
        }

        var identifier: String { base.identifier }
        var dimension: Int { base.dimension }
        var maxConcurrentRuns: Int { base.maxConcurrentRuns }

        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            let before = ProcessMemory.footprintMB()
            let start = Date()
            let vectors = try await base.embedDocuments(texts)
            let seconds = Date().timeIntervalSince(start)
            let after = ProcessMemory.footprintMB()
            lock.lock()
            events.append(Event(size: texts.count, seconds: seconds,
                                beforeMB: before, afterMB: after))
            lock.unlock()
            return vectors
        }

        func embedQuery(_ text: String) async throws -> [Float] {
            try await base.embedQuery(text)
        }

        func unload() async {
            await base.unload()
        }

        func snapshot() -> [Event] {
            lock.lock()
            defer { lock.unlock() }
            return events
        }
    }
}

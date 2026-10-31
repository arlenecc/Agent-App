import XCTest
@testable import BaseAgentCore

final class EmbeddingSchedulerTests: XCTestCase {

    private final class RecordingEmbedding: EmbeddingProvider, @unchecked Sendable {
        let identifier = "recording"
        let dimension = 1
        private let lock = NSLock()
        private(set) var batchSizes: [Int] = []
        private(set) var maxConcurrent = 0
        private var active = 0
        private let delay: UInt64

        init(delayNanoseconds: UInt64 = 0) {
            self.delay = delayNanoseconds
        }

        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            lock.lock()
            batchSizes.append(texts.count)
            active += 1
            maxConcurrent = max(maxConcurrent, active)
            lock.unlock()

            if delay > 0 {
                try? await Task.sleep(nanoseconds: delay)
            }

            lock.lock()
            active -= 1
            lock.unlock()
            return texts.map { text in
                [Float(text) ?? 0]
            }
        }

        func embedQuery(_ text: String) async throws -> [Float] {
            [Float(text) ?? 0]
        }
    }

    /// 记录底层实际收到的文本顺序，用于验证跨文件窗口交错。
    private final class OrderedRecordingEmbedding: EmbeddingProvider, @unchecked Sendable {
        let identifier = "ordered"
        let dimension = 1
        private let lock = NSLock()
        private(set) var texts: [String] = []
        private let delay: UInt64

        init(delayNanoseconds: UInt64) { self.delay = delayNanoseconds }

        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            lock.lock()
            self.texts.append(contentsOf: texts)
            lock.unlock()
            try? await Task.sleep(nanoseconds: delay)
            return texts.map { _ in [0] }
        }

        func embedQuery(_ text: String) async throws -> [Float] { [0] }
    }

    private final class CountingEmbedding: EmbeddingProvider, EmbeddingTokenCounting, @unchecked Sendable {
        let identifier = "counting"
        let dimension = 1
        let maxContentTokens = 128

        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map { _ in [0] }
        }

        func embedQuery(_ text: String) async throws -> [Float] {
            [0]
        }

        func makeTokenCounter() async -> (@Sendable (String) -> Int)? {
            { $0.count }
        }
    }

    /// 精确 token 计数 + 记录每批文本，用于验证「按 token 预算 + 长度排序组批」。
    private final class TokenBudgetRecordingEmbedding: EmbeddingProvider,
                                                      EmbeddingTokenCounting,
                                                      @unchecked Sendable {
        let identifier = "token-budget-recording"
        let dimension = 1
        let maxContentTokens = 4_096
        private let lock = NSLock()
        private(set) var batches: [[String]] = []

        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            lock.lock()
            batches.append(texts)
            lock.unlock()
            // 向量值 = 文本长度，便于验证结果按原始下标还原。
            return texts.map { [Float($0.count)] }
        }

        func embedQuery(_ text: String) async throws -> [Float] {
            [Float(text.count)]
        }

        func makeTokenCounter() async -> (@Sendable (String) -> Int)? {
            { $0.count }
        }
    }

    /// 按 token 预算（而不是块数）分组：长文本优先独占/先装满，
    /// 每个批次的 token 总量不超过预算，且结果顺序与输入一致。
    func testTokenBudgetGroupsByLengthAndRestoresInputOrder() async throws {
        let base = TokenBudgetRecordingEmbedding()
        let scheduler = AdaptiveEmbeddingScheduler(
            base: base,
            options: .init(minBatchSize: 1, maxBatchSize: 16,
                           initialBatchSize: 16, step: 1,
                           memoryLimitMB: 32_768, throughputTolerance: 0.5,
                           probeBatches: 16, probeSeconds: 30,
                           tokenBudgetMin: 1_024, tokenBudgetMax: 1_024,
                           initialTokenBudget: 1_024, tokenBudgetStep: 128),
            log: { _ in }
        )
        let lengths = [700, 100, 600, 50, 550, 300]
        let texts = lengths.map { String(repeating: "a", count: $0) }
        let vectors = try await scheduler.embedDocuments(texts)

        // 输出顺序必须与输入一致（内部按长度重排后要还原）。
        XCTAssertEqual(vectors.compactMap { $0.first.map(Int.init) }, lengths)
        let batches = base.batches
        // 700+300=1000，600+100+50=750，550 独占一批；每个子批都不超过 1024。
        XCTAssertEqual(batches.map { $0.map(\.count) }, [[700, 300], [600, 100, 50], [550]],
                       "长块优先，跳过装不下的项用更短的切片填满预算")
        for batch in batches {
            let tokens = batch.reduce(0) { $0 + $1.count }
            XCTAssertLessThanOrEqual(tokens, 1_024)
        }

        let metrics = scheduler.metricsSnapshot()
        XCTAssertEqual(metrics.batches, 3)
        XCTAssertEqual(metrics.texts, texts.count)
        XCTAssertEqual(metrics.tokens, lengths.reduce(0, +))
        XCTAssertEqual(metrics.observedMaxBatchTexts, 3)
        XCTAssertEqual(metrics.tokenBudget, 1_024)
        XCTAssertGreaterThan(metrics.tokensPerSecond, 0)
        XCTAssertGreaterThan(metrics.peakMemoryMB, 0)
    }

    /// 单条超过 token 预算时仍要能处理（独占一批），不能死循环或丢弃。
    func testTokenBudgetAllowsSingleOversizedText() async throws {
        let base = TokenBudgetRecordingEmbedding()
        let scheduler = AdaptiveEmbeddingScheduler(
            base: base,
            options: .init(minBatchSize: 1, maxBatchSize: 16,
                           initialBatchSize: 16, step: 1,
                           memoryLimitMB: 32_768, throughputTolerance: 0.5,
                           probeBatches: 16, probeSeconds: 30,
                           tokenBudgetMin: 1_024, tokenBudgetMax: 1_024,
                           initialTokenBudget: 1_024, tokenBudgetStep: 128),
            log: { _ in }
        )
        let texts = [String(repeating: "a", count: 2_000), "x", "y"]
        let vectors = try await scheduler.embedDocuments(texts)
        XCTAssertEqual(vectors.compactMap { $0.first.map(Int.init) }, [2_000, 1, 1])
        XCTAssertEqual(base.batches.map { $0.map(\.count) }, [[2_000], [1, 1]])
        XCTAssertEqual(scheduler.metricsSnapshot().tokens, 2_002)
    }

    /// ingest 拿到的是调度器而不是底层 provider，必须把窗口预算与精确计数器转发出去，
    /// 否则 bge 的 512 token 窗口在切片阶段完全失去保护。
    func testSchedulerForwardsTokenCountingCapability() async {
        let scheduler = AdaptiveEmbeddingScheduler(base: CountingEmbedding(), log: { _ in })
        XCTAssertEqual(scheduler.maxContentTokens, 128)
        let counter = await scheduler.makeTokenCounter()
        XCTAssertEqual(counter?("云计算"), 3)
    }

    func testSchedulerWithoutTokenCountingFallsBack() async {
        let scheduler = AdaptiveEmbeddingScheduler(base: RecordingEmbedding(), log: { _ in })
        XCTAssertEqual(scheduler.maxContentTokens, Int.max, "不支持的底层不限制切片大小")
        let counter = await scheduler.makeTokenCounter()
        XCTAssertNil(counter, "应回退到保守估算器")
    }

    func testSchedulerPreservesOrderAndBatchBounds() async throws {
        let base = RecordingEmbedding()
        let scheduler = AdaptiveEmbeddingScheduler(
            base: base,
            options: .init(minBatchSize: 2, maxBatchSize: 64,
                           initialBatchSize: 16, step: 2),
            log: { _ in }
        )
        let texts = (0..<37).map(String.init)
        let vectors = try await scheduler.embedDocuments(texts)

        XCTAssertEqual(vectors.count, texts.count)
        XCTAssertEqual(vectors.compactMap { $0.first.map(Int.init) }, Array(0..<37))
        XCTAssertFalse(base.batchSizes.isEmpty)
        XCTAssertTrue(base.batchSizes.allSatisfy { $0 >= 2 && $0 <= 64 },
                      "\(base.batchSizes)")
        XCTAssertEqual(base.batchSizes.reduce(0, +), texts.count)
    }

    /// 两个文件同时 embed 时，窗口之间必须让出信号量：
    /// 单个文件不能凭「先到先得」连续占满前几个窗口直到全部切片完成。
    func testSchedulerInterleavesFilesAcrossWindows() async throws {
        let base = OrderedRecordingEmbedding(delayNanoseconds: 30_000_000)
        let scheduler = AdaptiveEmbeddingScheduler(
            base: base,
            options: .init(minBatchSize: 1, maxBatchSize: 1,
                           initialBatchSize: 1, step: 1,
                           probeBatches: 16, probeSeconds: 30,
                           windowTexts: 1),
            log: { _ in })
        async let first = scheduler.embedDocuments(["a0", "a1", "a2"])
        async let second = scheduler.embedDocuments(["b0", "b1", "b2"])
        _ = try await (first, second)

        let prefix = Array(base.texts.prefix(4))
        XCTAssertTrue(prefix.contains { $0.hasPrefix("a") }, "\(prefix)")
        XCTAssertTrue(prefix.contains { $0.hasPrefix("b") }, "\(prefix)")
        var run = 1
        var maxRun = 1
        for i in 1..<prefix.count {
            let sameFile = prefix[i].prefix(1) == prefix[i - 1].prefix(1)
            run = sameFile ? run + 1 : 1
            maxRun = max(maxRun, run)
        }
        XCTAssertLessThanOrEqual(maxRun, 2,
                                 "单个文件不应连续占用多个窗口：\(prefix)")
    }

    /// 窗口切分后输出顺序与进度回调仍按整次调用的原始下标推进。
    func testSchedulerWindowsPreserveOrderAndProgress() async throws {
        let base = TokenBudgetRecordingEmbedding()
        let scheduler = AdaptiveEmbeddingScheduler(
            base: base,
            options: .init(minBatchSize: 1, maxBatchSize: 16,
                           initialBatchSize: 16, step: 1,
                           probeBatches: 16, probeSeconds: 30,
                           tokenBudgetMin: 4_096, tokenBudgetMax: 4_096,
                           initialTokenBudget: 4_096, tokenBudgetStep: 128,
                           windowTexts: 2),
            log: { _ in })
        let texts = ["a", "bb", "ccc", "dddd", "eeeee"]
        var progress: [(Int, Int)] = []
        let vectors = try await scheduler.embedDocuments(texts) { done, total in
            progress.append((done, total))
        }
        XCTAssertEqual(vectors.compactMap { $0.first.map(Int.init) }, [1, 2, 3, 4, 5])
        XCTAssertEqual(progress.map(\.0), [2, 4, 5])
        XCTAssertTrue(progress.allSatisfy { $0.1 == texts.count })
    }

    func testSchedulerSerializesConcurrentCalls() async throws {
        let base = RecordingEmbedding(delayNanoseconds: 20_000_000)
        let scheduler = AdaptiveEmbeddingScheduler(
            base: base,
            options: .init(minBatchSize: 2, maxBatchSize: 8,
                           initialBatchSize: 4, step: 2,
                           probeBatches: 1, probeSeconds: 0.01),
            log: { _ in }
        )
        async let first = scheduler.embedDocuments(["0", "1", "2", "3", "4", "5"])
        async let second = scheduler.embedDocuments(["6", "7", "8", "9", "10", "11"])
        _ = try await (first, second)

        XCTAssertEqual(base.maxConcurrent, 1)
    }

    /// 配置层可能传入越界值，Options 自己也要夹到安全范围。
    func testSchedulerOptionsNormalizeBounds() {
        let options = AdaptiveEmbeddingScheduler.Options(
            minBatchSize: 0, maxBatchSize: 1, initialBatchSize: 99, step: 0,
            memoryLimitMB: 1, throughputTolerance: 9,
            probeBatches: 0, probeSeconds: 0)
        XCTAssertEqual(options.minBatchSize, 1)
        XCTAssertEqual(options.maxBatchSize, 1)
        XCTAssertEqual(options.initialBatchSize, 1)
        XCTAssertEqual(options.step, 1)
        XCTAssertEqual(options.memoryLimitMB, 256)
        XCTAssertEqual(options.throughputTolerance, 0.5)
        XCTAssertEqual(options.probeBatches, 1)
        XCTAssertEqual(options.probeSeconds, 0.1)
    }
}

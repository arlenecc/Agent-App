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

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
}

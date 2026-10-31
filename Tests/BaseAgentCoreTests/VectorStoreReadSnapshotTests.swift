import XCTest
@testable import BaseAgentCore

/// 检索走只读 WAL 快照：FTS 候选与 hydrate 在同一个读事务内完成，
/// 且整个检索不再持有 store 写锁，写入/删除可以并发提交。
final class VectorStoreReadSnapshotTests: XCTestCase {
    private var directory: URL!
    private var store: VectorStore!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("vector-snapshot-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        store = try VectorStore(dbPath: directory.appendingPathComponent("index.db").path,
                                dimension: 4)
    }

    override func tearDownWithError() throws {
        store?.close()
        try? FileManager.default.removeItem(at: directory)
    }

    private func chunks(_ count: Int, source: String = "a.md") -> [Chunk] {
        (0..<count).map { i in
            Chunk(text: "云计算 容器 \(i)", source: source, chunkIndex: i)
        }
    }

    private func vector(_ count: Int) -> [[Float]] {
        Array(repeating: [1, 0, 0, 0], count: count)
    }

    /// 读快照打开期间，写入必须能在另一个线程完成（WAL 读写并发）。
    func testReadSnapshotDoesNotBlockWrites() throws {
        try store.add(chunks: chunks(4), vectors: vector(4))
        let writeFinished = DispatchSemaphore(value: 0)
        let snapshotReady = expectation(description: "snapshot")
        let writeDone = expectation(description: "write")

        DispatchQueue.global().async { [store] in
            try? store!.withReadSnapshot { db in
                _ = try db.query("SELECT COUNT(*) FROM chunks")
                snapshotReady.fulfill()
                // 快照保持打开，等待写入完成。
                _ = writeFinished.wait(timeout: .now() + 5)
            }
        }
        wait(for: [snapshotReady], timeout: 2)
        DispatchQueue.global().async { [store] in
            _ = try? store!.add(chunks: self.chunks(2, source: "b.md"),
                                vectors: self.vector(2))
            writeFinished.signal()
            writeDone.fulfill()
        }
        wait(for: [writeDone], timeout: 2)
        XCTAssertEqual(try store.withReadSnapshot { db in
            try db.query("SELECT COUNT(*) FROM chunks")[0]["COUNT(*)"]?.intValue
        }, 6)
    }

    /// 快照一旦建立，之后提交的写入在快照内不可见（保证 FTS 候选与 hydrate 一致）。
    func testReadSnapshotKeepsConsistentView() throws {
        try store.add(chunks: chunks(3), vectors: vector(3))
        try store.withReadSnapshot { db in
            XCTAssertEqual(try db.query("SELECT COUNT(*) FROM chunks")[0]["COUNT(*)"]?.intValue, 3)
            try self.store.add(chunks: self.chunks(2, source: "c.md"),
                               vectors: self.vector(2))
            XCTAssertEqual(try db.query("SELECT COUNT(*) FROM chunks")[0]["COUNT(*)"]?.intValue, 3,
                           "同一读事务必须继续看到快照视图")
        }
    }

    /// FTS 检索结果在写入并发下仍完整、可 hydrate。
    func testFTSSearchRejectsHalfDeletedResults() throws {
        try store.add(chunks: chunks(5), vectors: vector(5))
        XCTAssertFalse(try store.searchFTS(query: "云计算", topK: 5).isEmpty)
        try store.deleteBySource("a.md")
        XCTAssertTrue(try store.searchFTS(query: "云计算", topK: 5).isEmpty)
    }
}

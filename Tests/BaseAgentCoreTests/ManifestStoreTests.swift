import XCTest
@testable import BaseAgentCore

/// Manifest 增量写入：journal 追加、加载回放、压实与清空。
final class ManifestStoreTests: XCTestCase {
    private var directory: URL!
    private var manifestPath: String { directory.appendingPathComponent("manifest.json").path }
    private var journalPath: String { manifestPath + ".journal" }

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("manifest-store-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func entry(_ digest: String,
                       complete: Bool = true) -> RAGManifestEntry {
        RAGManifestEntry(mtime: 1, size: 10, fingerprint: "fp", digest: digest,
                         docName: "doc.md", mdPath: "/tmp/doc.md",
                         ingestSignature: "sig", chunkCount: 3, complete: complete,
                         duplicateOf: nil)
    }

    func testMergeAppendsJournalAndLoadReplays() throws {
        let store = ManifestStore(path: manifestPath)
        store.merge(updated: ["a": entry("da")], removed: [])
        store.merge(updated: ["b": entry("db")], removed: [])

        let loaded = store.load()
        XCTAssertEqual(loaded["a"]?.digest, "da")
        XCTAssertEqual(loaded["b"]?.digest, "db")
        XCTAssertFalse(FileManager.default.fileExists(atPath: manifestPath),
                       "增量写入不应每次都重写完整 JSON")
        XCTAssertTrue(FileManager.default.fileExists(atPath: journalPath))
    }

    func testRemovedEntriesReplayFromJournal() throws {
        let store = ManifestStore(path: manifestPath)
        store.merge(updated: ["a": entry("da")], removed: [])
        store.merge(updated: [:], removed: ["a"])
        XCTAssertTrue(store.load().isEmpty)
    }

    func testTruncatedJournalTailIsIgnored() throws {
        let store = ManifestStore(path: manifestPath)
        store.merge(updated: ["a": entry("da")], removed: [])

        // 真实追加格式以换行开头：崩溃只会留下一条独立的半行记录。
        let handle = try FileHandle(forWritingTo: URL(fileURLWithPath: journalPath))
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(("\n" + #"{"updated":"#).utf8))
        try handle.close()

        XCTAssertEqual(store.load()["a"]?.digest, "da")
    }

    func testCompactWritesBaseAndRemovesJournal() throws {
        let store = ManifestStore(path: manifestPath)
        store.merge(updated: ["a": entry("da")], removed: [])
        store.merge(updated: ["b": entry("db")], removed: ["a"])

        store.compact()

        XCTAssertFalse(FileManager.default.fileExists(atPath: journalPath))
        let data = try Data(contentsOf: URL(fileURLWithPath: manifestPath))
        let manifest = try JSONDecoder().decode(RAGManifest.self, from: data)
        XCTAssertEqual(manifest.files["b"]?.digest, "db")
        XCTAssertNil(manifest.files["a"])
        XCTAssertEqual(store.load()["b"]?.digest, "db")
    }

    func testClearRemovesBaseAndJournal() throws {
        let store = ManifestStore(path: manifestPath)
        store.merge(updated: ["a": entry("da")], removed: [])
        store.compact()
        store.merge(updated: ["b": entry("db")], removed: [])

        store.clear()

        XCTAssertFalse(FileManager.default.fileExists(atPath: manifestPath))
        XCTAssertFalse(FileManager.default.fileExists(atPath: journalPath))
        XCTAssertTrue(store.load().isEmpty)
    }

    /// 压实后继续追加的记录必须叠加在新基线之上。
    func testMergeAfterCompactReplaysOnTopOfBase() throws {
        let store = ManifestStore(path: manifestPath)
        store.merge(updated: ["a": entry("da")], removed: [])
        store.compact()
        store.merge(updated: ["b": entry("db")], removed: ["a"])

        let loaded = store.load()
        XCTAssertEqual(loaded["b"]?.digest, "db")
        XCTAssertNil(loaded["a"])
    }
}

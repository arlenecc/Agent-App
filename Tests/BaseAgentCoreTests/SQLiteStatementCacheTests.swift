import XCTest
@testable import BaseAgentCore

/// per-connection statement cache：同一 SQL 反复执行不再 prepare/finalize，
/// 且提前结束的 stream 必须 reset，避免同一连接上的事务无法提交。
final class SQLiteStatementCacheTests: XCTestCase {
    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("statement-cache-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDir)
    }

    private func makeDatabase() throws -> SQLiteDatabase {
        try SQLiteDatabase(path: tempDir.appendingPathComponent("test.sqlite").path)
    }

    func testRepeatedSQLReusesCachedStatement() throws {
        let db = try makeDatabase()
        try db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, value TEXT)")
        let afterDDL = db.statementCacheSize
        for index in 0..<40 {
            try db.execute("INSERT INTO items (value) VALUES (?)", [.text("v\(index)")])
        }
        // 首次执行某条 SQL 会 prepare 并放入缓存，之后的 39 次都复用。
        XCTAssertEqual(db.statementCacheSize, afterDDL + 1,
                       "同一 INSERT 应复用缓存语句，不再新增 prepare")

        let selectSQL = "SELECT id, value FROM items WHERE id = ?"
        for _ in 0..<20 {
            let row = try db.query(selectSQL, [.int(1)]).first
            XCTAssertEqual(row?["value"]?.stringValue, "v0")
        }
        XCTAssertEqual(db.statementCacheSize, afterDDL + 2,
                       "同一 SELECT 只应准备一次")
    }

    func testStreamEarlyExitDoesNotBlockTransactionCommit() throws {
        let db = try makeDatabase()
        try db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, value TEXT)")
        // 修复前：stream 提前 break 仍停在 SQLITE_ROW，未 reset 的缓存语句
        // 会让 COMMIT 报 "cannot commit transaction - SQL statements in progress"。
        try db.transaction {
            try db.execute("INSERT INTO items (value) VALUES (?)", [.text("a")])
            try db.stream("SELECT id, value FROM items ORDER BY id") { _ in false }
            try db.execute("INSERT INTO items (value) VALUES (?)", [.text("b")])
        }
        let rows = try db.query("SELECT value FROM items ORDER BY id")
        XCTAssertEqual(rows.compactMap { $0["value"]?.stringValue }, ["a", "b"])
    }

    func testCacheStaysWithinBound() throws {
        let db = try makeDatabase()
        try db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)")
        // 每条 SQL 不同，缓存应被限制在上限内，且之后仍能正常查询。
        for index in 0..<100 {
            _ = try db.query("SELECT \(index) AS value")
        }
        XCTAssertLessThanOrEqual(db.statementCacheSize, 64)
        XCTAssertEqual(try db.query("SELECT 42 AS value").first?["value"]?.intValue, 42)
    }
}

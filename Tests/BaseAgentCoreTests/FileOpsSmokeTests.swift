import XCTest
@testable import BaseAgentCore

/// 文件工具行为冒烟测试：write → read 往返、modify 替换、路径护栏、截断。
final class FileOpsSmokeTests: XCTestCase {
    private var root: String!

    override func setUp() {
        super.setUp()
        root = NSTemporaryDirectory() + "fileops-smoke-\(UUID().uuidString.prefix(6))"
        try? FileManager.default.createDirectory(atPath: root, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(atPath: root)
        root = nil
        super.tearDown()
    }

    private var config: AgentConfig {
        var config = AgentConfig()
        config.workspace = root
        return config
    }

    private func write(_ path: String, _ content: String) async throws -> ToolResult {
        try await FileWriteTool(config: config, callbacks: nil)
            .run(["path": .string(path), "content": .string(content)])
    }

    private func read(_ path: String) async throws -> ToolResult {
        try await FileReadTool(config: config, callbacks: nil)
            .run(["path": .string(path)])
    }

    private func modify(_ path: String, old: String, new: String,
                        all: Bool = false) async throws -> ToolResult {
        var args: [String: JSONValue] = [
            "path": .string(path), "old": .string(old), "new": .string(new),
        ]
        if all { args["replace_all"] = .string("true") }
        return try await FileModifyTool(config: config, callbacks: nil).run(args)
    }

    func testWriteReadRoundtripWithUnicode() async throws {
        let content = "# 标题\n\n- 中文条目 1\n- émoji: ✅\n- code: `x = 1`\n"
        let result = try await write("notes/中文笔记.md", content)
        XCTAssertTrue(result.success, "write 失败: \(result.error)")

        let back = try await read("notes/中文笔记.md")
        XCTAssertTrue(back.success)
        XCTAssertTrue(back.output.contains("中文条目"), "内容应往返一致: \(back.output)")
    }

    func testWriteCreatesSubdirectories() async throws {
        let result = try await write("a/b/c/deep.txt", "deep")
        XCTAssertTrue(result.success)
        XCTAssertTrue(FileManager.default.fileExists(atPath: root + "/a/b/c/deep.txt"))
    }

    func testModifyReplacesFirstOccurrenceOnly() async throws {
        _ = try await write("list.txt", "alpha\nbeta\nalpha\n")
        let result = try await modify("list.txt", old: "alpha", new: "ALPHA")
        XCTAssertTrue(result.success, "modify 失败: \(result.error)")
        let back = try await read("list.txt")
        XCTAssertTrue(back.output.contains("ALPHA\nbeta\nalpha"), "默认只替换第一处")
    }

    func testModifyReplaceAll() async throws {
        _ = try await write("list.txt", "alpha beta alpha\n")
        let result = try await modify("list.txt", old: "alpha", new: "X", all: true)
        XCTAssertTrue(result.success)
        let back = try await read("list.txt")
        XCTAssertTrue(back.output.contains("X beta X"))
        XCTAssertFalse(back.output.contains("alpha"))
    }

    func testModifyMissingOldFails() async throws {
        _ = try await write("list.txt", "hello world\n")
        let result = try await modify("list.txt", old: "not-present", new: "x")
        XCTAssertFalse(result.success, "old 未命中应报错")
        XCTAssertTrue(result.error.contains("not found"),
                      "错误信息应包含线索: \(result.error)")
    }

    func testReadMissingFileFails() async throws {
        let result = try await read("nope/missing.txt")
        XCTAssertFalse(result.success)
    }

    /// file_read 固定 20 万字符上限（不可参数化）：超限文件应截断并提示总量。
    func testReadTruncationReportsBound() async throws {
        let long = String(repeating: "abcdefghij", count: 25_000) // 25 万字符
        _ = try await write("long.txt", long)
        let result = try await read("long.txt")
        XCTAssertTrue(result.success)
        XCTAssertTrue(result.output.contains("仅显示前") || result.output.contains("截断"),
                      "超限截断应有提示: \(result.output.suffix(200))")
    }

    func testOutsideWorkspacePathsRejected() async throws {
        for bad in ["../escape.txt", "/tmp/outside-\(UUID().uuidString.prefix(4)).txt",
                    "/etc/passwd"] {
            let result = try await write(bad, "nope")
            XCTAssertFalse(result.success, "路径 \(bad) 应被拒绝")
        }
        let modifyResult = try await modify("../outside.txt", old: "a", new: "b")
        XCTAssertFalse(modifyResult.success, "modify 越界也应被拒绝")
        let readResult = try await read("/etc/passwd")
        XCTAssertFalse(readResult.success, "read 越界也应被拒绝")
    }
}

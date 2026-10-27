import XCTest
@testable import BaseAgentCore

final class MarkdownConversionServiceTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-convert-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    func testSingleTextFileWritesMarkdown() async throws {
        let source = root.appendingPathComponent("note.txt")
        try "第一段文字。\n\n第二段文字。".write(to: source, atomically: true, encoding: .utf8)
        let output = root.appendingPathComponent("out")
        var progress: [MarkdownConversionProgress] = []

        let report = try await MarkdownConversionService().convert(
            request: MarkdownConversionRequest(sourceURL: source,
                                               outputDirectory: output,
                                               config: AgentConfig())
        ) { progress.append($0) }

        XCTAssertFalse(report.wasCancelled)
        XCTAssertEqual(report.results.count, 1)
        XCTAssertEqual(report.succeededCount, 1)
        XCTAssertEqual(report.failedCount, 0)

        let markdown = output.appendingPathComponent("note.md")
        XCTAssertTrue(FileManager.default.fileExists(atPath: markdown.path))
        let text = try String(contentsOf: markdown, encoding: .utf8)
        XCTAssertTrue(text.contains("第一段文字"))
        XCTAssertTrue(text.contains("第二段文字"))

        XCTAssertEqual(progress.last?.phase, .finished)
        XCTAssertEqual(progress.last?.fileTotal, 1)
        XCTAssertEqual(progress.last?.completedFiles, 1)
        XCTAssertEqual(progress.last?.overallFraction, 1)
    }

    func testDirectoryKeepsRelativeStructureAndSkipsOutputDirectory() async throws {
        let input = root.appendingPathComponent("input")
        let nested = input.appendingPathComponent("sub")
        let output = input.appendingPathComponent("out")
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try "alpha".write(to: input.appendingPathComponent("a.txt"),
                          atomically: true, encoding: .utf8)
        try "<html><body><h1>Beta</h1><p>body text</p></body></html>"
            .write(to: nested.appendingPathComponent("b.html"),
                   atomically: true, encoding: .utf8)
        // 上次转换结果位于输出目录内，本次不应再次处理。
        try "old output".write(to: output.appendingPathComponent("old.txt"),
                               atomically: true, encoding: .utf8)

        let report = try await MarkdownConversionService().convert(
            request: MarkdownConversionRequest(sourceURL: input,
                                               outputDirectory: output,
                                               config: AgentConfig()))

        XCTAssertEqual(report.results.count, 2)
        XCTAssertEqual(report.succeededCount, 2)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: output.appendingPathComponent("a.md").path))
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: output.appendingPathComponent("sub/b.md").path))
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: output.appendingPathComponent("old.md").path))
        XCTAssertTrue(try String(contentsOf: output.appendingPathComponent("sub/b.md"),
                                 encoding: .utf8).contains("Beta"))
    }

    func testDuplicateBaseNamesGetSuffix() async throws {
        let input = root.appendingPathComponent("input")
        let output = root.appendingPathComponent("out")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        try "text source".write(to: input.appendingPathComponent("a.txt"),
                                atomically: true, encoding: .utf8)
        try "<html><body><h1>html source</h1></body></html>"
            .write(to: input.appendingPathComponent("a.html"),
                   atomically: true, encoding: .utf8)

        let report = try await MarkdownConversionService().convert(
            request: MarkdownConversionRequest(sourceURL: input,
                                               outputDirectory: output,
                                               config: AgentConfig()))

        XCTAssertEqual(report.results.count, 2)
        XCTAssertEqual(report.succeededCount, 2)
        let first = output.appendingPathComponent("a.md")
        let second = output.appendingPathComponent("a-2.md")
        XCTAssertTrue(FileManager.default.fileExists(atPath: first.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: second.path))
        let contents = try String(contentsOf: first, encoding: .utf8)
            + String(contentsOf: second, encoding: .utf8)
        XCTAssertTrue(contents.contains("text source"))
        XCTAssertTrue(contents.contains("html source"))
    }

    func testFailureDoesNotAbortBatch() async throws {
        let input = root.appendingPathComponent("input")
        let output = root.appendingPathComponent("out")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        try "valid text".write(to: input.appendingPathComponent("ok.txt"),
                               atomically: true, encoding: .utf8)
        // 非法 zip：模拟损坏的 Office 文件。
        try "not a zip".write(to: input.appendingPathComponent("broken.docx"),
                             atomically: true, encoding: .utf8)

        let report = try await MarkdownConversionService().convert(
            request: MarkdownConversionRequest(sourceURL: input,
                                               outputDirectory: output,
                                               config: AgentConfig()))

        XCTAssertEqual(report.results.count, 2)
        XCTAssertEqual(report.succeededCount, 1)
        XCTAssertEqual(report.failedCount, 1)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: output.appendingPathComponent("ok.md").path))
        XCTAssertNotNil(report.results.first(where: {
            $0.source.lastPathComponent == "broken.docx"
        })?.error)
    }

    func testUnsupportedSingleFileThrows() async throws {
        let source = root.appendingPathComponent("image.gif")
        try "gif".write(to: source, atomically: true, encoding: .utf8)

        do {
            _ = try await MarkdownConversionService().convert(
                request: MarkdownConversionRequest(
                    sourceURL: source,
                    outputDirectory: root.appendingPathComponent("out"),
                    config: AgentConfig()))
            XCTFail("应当拒绝不支持的格式")
        } catch let error as MarkdownConversionError {
            guard case .sourceNotSupported = error else {
                return XCTFail("错误类型不符：\(error)")
            }
        }
    }

    func testDirectoryWithoutSupportedFilesThrows() async throws {
        let input = root.appendingPathComponent("input")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        try "gif".write(to: input.appendingPathComponent("a.gif"),
                        atomically: true, encoding: .utf8)

        do {
            _ = try await MarkdownConversionService().convert(
                request: MarkdownConversionRequest(
                    sourceURL: input,
                    outputDirectory: root.appendingPathComponent("out"),
                    config: AgentConfig()))
            XCTFail("没有可转换文件时应报错")
        } catch let error as MarkdownConversionError {
            guard case .noSupportedFiles = error else {
                return XCTFail("错误类型不符：\(error)")
            }
        }
    }
}

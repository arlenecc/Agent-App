import XCTest
import ZIPFoundation
@testable import BaseAgentCore

final class OfficeAndHTMLTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-office-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func testDocxParagraphsHeadingsAndTables() throws {
        let url = directory.appendingPathComponent("sample.docx")
        let documentXML = """
        <?xml version="1.0"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body>
            <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>项目概述</w:t></w:r></w:p>
            <w:p><w:r><w:t>这是一个 DOCX 段落。</w:t></w:r><w:r><w:t>第二段文字。</w:t></w:r></w:p>
            <w:tbl>
              <w:tr><w:tc><w:p><w:r><w:t>列A</w:t></w:r></w:p></w:tc>
                    <w:tc><w:p><w:r><w:t>列B</w:t></w:r></w:p></w:tc></w:tr>
              <w:tr><w:tc><w:p><w:r><w:t>1</w:t></w:r></w:p></w:tc>
                    <w:tc><w:p><w:r><w:t>2</w:t></w:r></w:p></w:tc></w:tr>
            </w:tbl>
          </w:body>
        </w:document>
        """
        try makeArchive(at: url, files: [
            ("word/document.xml", documentXML),
            ("[Content_Types].xml", "<?xml version=\"1.0\"?><Types/>"),
        ])
        let parsed = try awaitSync {
            try await OfficeParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("# 项目概述"))
        XCTAssertTrue(parsed.text.contains("这是一个 DOCX 段落"))
        XCTAssertTrue(parsed.text.contains("| 列A | 列B |"))
        XCTAssertEqual(parsed.metadata["engine"], "docx")
    }

    func testHTMLToMarkdown() throws {
        let url = directory.appendingPathComponent("page.html")
        let html = """
        <html><head><title>Demo</title><script>alert(1)</script></head>
        <body><h1>标题一</h1><p>正文 <b>加粗</b>。</p>
        <ul><li>条目一</li><li>条目二</li></ul>
        <table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>
        </body></html>
        """
        try html.write(to: url, atomically: true, encoding: .utf8)
        let parsed = try awaitSync {
            try await HTMLParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("# 标题一"))
        XCTAssertTrue(parsed.text.contains("加粗"))
        XCTAssertTrue(parsed.text.contains("条目一"))
        XCTAssertTrue(parsed.text.contains("| A | B |"))
        XCTAssertFalse(parsed.text.contains("alert(1)"))
    }

    func testBoldOnlyParagraphBecomesHeading() throws {
        let url = directory.appendingPathComponent("calibre.html")
        let html = """
        <html><head><title>Demo</title></head><body>
        <p class="c1"><b class="calibre1">这是金钱的使用价值</b></p>
        <p>正文第一段，讲述金钱的使用价值与收藏价值。</p>
        <p><b>强调</b>开头的正文段落，后面还有更多内容需要继续阅读。</p>
        <div><b>独立加粗的短句</b></div>
        </body></html>
        """
        try html.write(to: url, atomically: true, encoding: .utf8)
        let parsed = try awaitSync {
            try await HTMLParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("### 这是金钱的使用价值"),
                      "纯 <b> 段落应转为标题")
        XCTAssertTrue(parsed.outline.contains("这是金钱的使用价值"),
                      "纯 <b> 段落应进入大纲")
        XCTAssertFalse(parsed.outline.contains("强调"),
                       "段首强调不是标题")
        XCTAssertTrue(parsed.text.contains("**强调**开头的正文段落"),
                      "段首强调应保留为正文")
        XCTAssertTrue(parsed.outline.contains("独立加粗的短句"),
                      "纯 <b> 的 div 也应进入大纲")
    }

    // MARK: helpers

    private func awaitSync<T>(_ body: @escaping () async throws -> T) throws -> T {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<T, Error>!
        Task {
            do { result = .success(try await body()) }
            catch { result = .failure(error) }
            semaphore.signal()
        }
        _ = semaphore.wait(timeout: .now() + 60)
        return try result.get()
    }

    private func makeArchive(at url: URL, files: [(String, String)]) throws {
        guard let archive = try Archive(url: url, accessMode: .create) else {
            throw AgentError.parse("无法创建归档")
        }
        for (path, content) in files {
            let data = Data(content.utf8)
            try archive.addEntry(with: path, type: .file,
                                 uncompressedSize: Int64(data.count),
                                 compressionMethod: .deflate,
                                 provider: { position, size in
                data.subdata(in: Int(position)..<Int(position) + size)
            })
        }
    }
}

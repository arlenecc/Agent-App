import XCTest
@testable import BaseAgentCore

/// web_scan 的列表页/SPA 动态加载检测提示。
final class WebScanHintTests: XCTestCase {

    func testHintTriggeredForShortBodyWithLoader() {
        let html = """
        <html><head><title>Search Jobs</title>
        <script src="/js/app.js"></script></head>
        <body><form id="MainSearchForm"><input id="txtWhatKeyword"></form>
        <div class="jobs-loader">Loading jobs please wait...</div>
        <script>$.ajax({url: "/api/SearchIndex/GetSearch"});</script>
        </body></html>
        """
        let content = ToolSupport.htmlToText(html)
        let hint = ToolSupport.dynamicLoadingHint(html: html, extracted: content)
        XCTAssertTrue(hint.contains("web_scan 提示"))
        XCTAssertTrue(hint.contains("加载占位"))
        XCTAssertTrue(hint.contains("AJAX"))
    }

    func testNoHintForLongBody() {
        let html = "<html><body><p>\(String(repeating: "正文内容。", count: 200))"
            + "<script>fetch('/api/x')</script></body></html>"
        let content = ToolSupport.htmlToText(html)
        XCTAssertGreaterThan(content.count, 300)
        XCTAssertEqual(ToolSupport.dynamicLoadingHint(html: html, extracted: content), "")
    }

    func testNoHintForShortStaticPage() {
        // 正文短但没有 AJAX/框架特征：纯静态小页面，不应提示。
        let html = "<html><body><p>OK</p></body></html>"
        let content = ToolSupport.htmlToText(html)
        XCTAssertEqual(ToolSupport.dynamicLoadingHint(html: html, extracted: content), "")
    }

    func testHintForReactMountPoint() {
        let html = """
        <html><head><script src="/bundle.js"></script></head>
        <body><div id="root"></div></body></html>
        """
        let content = ToolSupport.htmlToText(html)
        let hint = ToolSupport.dynamicLoadingHint(html: html, extracted: content)
        XCTAssertTrue(hint.contains("前端框架挂载点"))
    }

    func testHintContainsDataEndpointClues() {
        // applytoeducation 形态：AJAX 端点 + Azure Search JS 引用。
        let html = """
        <html><body><div class="jobs-loader">Loading...</div>
        <script>$.ajax({url: "/api/SearchIndex/GetSearch?SearchText=job,*"});</script>
        <script src="/jscripts/azure-search-ui.min.js"></script>
        </body></html>
        """
        let content = ToolSupport.htmlToText(html)
        let hint = ToolSupport.dynamicLoadingHint(html: html, extracted: content)
        XCTAssertTrue(hint.contains("/api/SearchIndex/GetSearch"))
        XCTAssertTrue(hint.contains("azure-search-ui.min.js"))
        XCTAssertTrue(hint.contains("已定位的数据源线索"))
    }

    func testIframeOnlyPageTriggersHint() {
        // 内容全在 iframe 里，主文档是壳：也应触发提示并列出 iframe。
        let html = """
        <html><body><h1>Welcome</h1>
        <iframe src="/Applicant/JobsList.aspx"></iframe></body></html>
        """
        let content = ToolSupport.htmlToText(html)
        XCTAssertLessThan(content.trimmingCharacters(in: .whitespaces).count, 300)
        let hint = ToolSupport.dynamicLoadingHint(html: html, extracted: content)
        XCTAssertTrue(hint.contains("iframe"))
        XCTAssertTrue(hint.contains("/Applicant/JobsList.aspx"))
    }

    func testExtractIframeSourcesDedupAndLimit() {
        let html = """
        <iframe src="/a.html"></iframe>
        <iframe src="/a.html"></iframe>
        <iframe src="/b.html"></iframe>
        <iframe src="/c.html"></iframe>
        """
        XCTAssertEqual(ToolSupport.extractIframeSources(html: html, limit: 2),
                       ["/a.html", "/b.html"])
    }

}

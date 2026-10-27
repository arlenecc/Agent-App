import XCTest
@testable import BaseAgentCore

/// web_scan 对动态页外部 JS 的筛选：只抓同源、优先数据/搜索脚本。
final class WebScriptSourceTests: XCTestCase {

    private let base = URL(string: "https://network.applytoeducation.com/Applicant/AttSearchex.aspx?DirectLink=1")!

    func testSelectsDataScriptsAndFiltersCrossOrigin() {
        let html = """
        <html><body>
        <script src="/jScripts/jquery/js/jquery.js"></script>
        <script src="/jScripts/bootstrap.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@finsweet/attributes-cmsfilter@1/cmsfilter.js"></script>
        <script src="/jscripts/azure-search-ui-settings.min.js"></script>
        <script src="/jscripts/azure-search-ui.min.js"></script>
        <script src="/jscripts/applications.js"></script>
        </body></html>
        """
        let urls = ToolSupport.selectScriptSourcesForInspection(
            html: html, baseURL: base, limit: 2)
        XCTAssertEqual(urls.map(\.path), [
            "/jscripts/azure-search-ui-settings.min.js",
            "/jscripts/azure-search-ui.min.js",
        ])
    }

    func testSelectionDeduplicatesAndResolvesRelativeURLs() {
        let html = """
        <script src="../jscripts/azure-search-ui-settings.min.js"></script>
        <script src="https://network.applytoeducation.com/jscripts/azure-search-ui-settings.min.js"></script>
        <script src="/jscripts/azure-search-ui.min.js"></script>
        """
        let urls = ToolSupport.selectScriptSourcesForInspection(
            html: html, baseURL: base, limit: 5)
        XCTAssertEqual(urls.count, 2)
        XCTAssertTrue(urls.allSatisfy { $0.host == "network.applytoeducation.com" })
        XCTAssertEqual(urls.first?.path, "/jscripts/azure-search-ui-settings.min.js")
    }

    func testSelectionFallsBackToFirstSameOriginScript() {
        let html = """
        <script src="https://cdn.example.com/lib.js"></script>
        <script src="/js/site.js"></script>
        <script src="/js/theme.js"></script>
        """
        let urls = ToolSupport.selectScriptSourcesForInspection(
            html: html, baseURL: base, limit: 1)
        XCTAssertEqual(urls.map(\.path), ["/js/site.js"])
    }
}

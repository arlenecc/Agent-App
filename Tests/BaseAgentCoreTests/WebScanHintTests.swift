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

    func testHintTriggeredForLongBodyWithLoadingPlaceholderAndAjax() {
        // 真实招聘站形态：导航/页脚/营销文案让正文远超过 300 字符，
        // 但正文主体仍是「Loading jobs please wait...」占位。
        let chrome = String(repeating: "Navigation and footer content. ", count: 30)
        let html = """
        <html><head><script src="/jscripts/azure-search-ui-settings.min.js"></script></head>
        <body><nav>\(chrome)</nav>
        <h1>Search Jobs</h1>
        <div id="job-list"><div class="jobs-loader">Loading jobs please wait...</div></div>
        <footer>\(chrome)</footer>
        <script>$.ajax({url: "/api/searchip/getcountry"});</script>
        </body></html>
        """
        let content = ToolSupport.htmlToText(html)
        XCTAssertGreaterThan(content.count, 300)
        let hint = ToolSupport.dynamicLoadingHint(html: html, extracted: content)
        XCTAssertTrue(hint.contains("web_scan 提示"))
        XCTAssertTrue(hint.contains("加载占位"))
        XCTAssertTrue(hint.contains("AJAX"))
        XCTAssertTrue(hint.contains("主要是导航/页脚"))
    }

    func testNoHintForLongBodyWithLoadingWordOnly() {
        // 长静态页面只是碰巧出现 "please wait" 文案：没有 loader/AJAX，
        // 不应误判为动态加载。
        let body = String(repeating: "Static article body. ", count: 30)
        let html = "<html><body><p>Please wait while the document loads.</p>"
            + "<p>\(body)</p></body></html>"
        let content = ToolSupport.htmlToText(html)
        XCTAssertGreaterThan(content.count, 300)
        XCTAssertEqual(ToolSupport.dynamicLoadingHint(html: html, extracted: content), "")
    }

    func testEndpointHintsIgnoreStaticAssets() {
        // jquery 路径里的 "query"、引用的 .css/.js 都被误当接口过。
        let html = """
        <html><head>
        <link rel="stylesheet" href="/Styles/ate-jquery-ui.css">
        <script src="/jScripts/jquery/js/jquery.js"></script>
        <script src="/jscripts/azure-search-ui.min.js"></script>
        </head><body>
        <script>$.ajax({url: "/api/SearchIndex/GetSearch"});</script>
        </body></html>
        """
        let hints = ToolSupport.extractAPIEndpointHints(html: html)
        XCTAssertTrue(hints.contains("/api/SearchIndex/GetSearch"))
        XCTAssertFalse(hints.contains { $0.lowercased().contains("jquery") })
        XCTAssertFalse(hints.contains { $0.lowercased().hasSuffix(".css") })
        XCTAssertFalse(hints.contains { $0.lowercased().hasSuffix(".js") })
    }

    func testHintKeepsApiEndpointsWhenHtmlHasAssetNoise() {
        // 静态资源噪声占满候选名额时，外部 JS 里的真实接口仍必须出现。
        let chrome = String(repeating: "Navigation and footer content. ", count: 30)
        let html = """
        <html><head>
        <link rel="stylesheet" href="/Styles/ate-jquery-ui.css">
        <link rel="stylesheet" href="/Styles/ate-jquery-ui.theme.css">
        <script src="/jScripts/jquery/js/jquery.js"></script>
        <script src="/jScripts/jquery/js/jquery-ui.js"></script>
        <script src="/jScripts/jquery/js/jquery-migrate.js"></script>
        <script src="/jscripts/azure-search-ui.min.js"></script>
        <script src="/jscripts/azure-search-ui-settings.min.js"></script>
        <script src="/jScripts/search-helpers.js"></script>
        </head><body><nav>\(chrome)</nav>
        <div class="jobs-loader">Loading jobs please wait...</div>
        <script>$.ajax({url: "/api/searchip/getcountry"});</script>
        </body></html>
        """
        let js = #"settings.api_request.search_submit = "/api/SearchIndex/GetSearch";"#
        let hint = ToolSupport.dynamicLoadingHint(
            html: html,
            extracted: ToolSupport.htmlToText(html),
            externalScriptTexts: [js],
            inspectedScriptURLs: ["/jscripts/azure-search-ui-settings.min.js"])
        XCTAssertTrue(hint.contains("/api/SearchIndex/GetSearch"))
    }

    func testHintMergesEndpointsFromExternalScripts() {
        let chrome = String(repeating: "Navigation and footer content. ", count: 30)
        let html = """
        <html><head><script src="/jscripts/azure-search-ui-settings.min.js"></script></head>
        <body><nav>\(chrome)</nav>
        <div class="jobs-loader">Loading jobs please wait...</div>
        <script>$.ajax({url: "/api/searchip/getcountry"});</script>
        </body></html>
        """
        let js = """
        settings.api_request.search_submit = "/api/SearchIndex/GetSearch";
        settings.api_request.facet_submit = "/api/SearchIndex/Search";
        """
        let hint = ToolSupport.dynamicLoadingHint(
            html: html,
            extracted: ToolSupport.htmlToText(html),
            externalScriptTexts: [js],
            inspectedScriptURLs: ["/jscripts/azure-search-ui-settings.min.js"])
        XCTAssertTrue(hint.contains("/api/SearchIndex/GetSearch"))
        XCTAssertTrue(hint.contains("/api/SearchIndex/Search"))
        XCTAssertTrue(hint.contains("已扫描 JS"))
        XCTAssertTrue(hint.contains("azure-search-ui-settings.min.js"))
    }

}

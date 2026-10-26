import Foundation
import ImageIO
import ZIPFoundation
import AEXML
import SwiftSoup

// MARK: - 共享 ZIP / XML / 路径工具（OfficeParser 复用）

/// 归档读取：统一把 ZIPFoundation 的失败转换为 AgentError。
enum RAGArchiveReader {
    static func open(_ url: URL) throws -> Archive {
        do {
            return try Archive(url: url, accessMode: .read)
        } catch {
            throw AgentError.parse("无法解包文件 \(url.lastPathComponent)：\(error.localizedDescription)")
        }
    }

    static func data(_ archive: Archive, path: String) -> Data? {
        let normalized = path.hasPrefix("/") ? String(path.dropFirst()) : path
        guard !normalized.isEmpty, let entry = archive[normalized], entry.type == .file else {
            return nil
        }
        var data = Data()
        do {
            _ = try archive.extract(entry) { data.append($0) }
            return data
        } catch {
            return nil
        }
    }

    static func filePaths(_ archive: Archive) -> [String] {
        var paths: [String] = []
        for entry in archive where entry.type == .file {
            paths.append(entry.path)
        }
        return paths
    }
}

/// AEXML 辅助：按本地名（忽略命名空间前缀）查找元素/属性。
enum RAGXMLHelper {
    static func localName(_ name: String) -> String {
        name.split(separator: ":").last.map(String.init) ?? name
    }

    static func children(_ element: AEXMLElement, _ local: String) -> [AEXMLElement] {
        let target = local.lowercased()
        return element.children.filter { localName($0.name).lowercased() == target }
    }

    static func firstChild(_ element: AEXMLElement, _ local: String) -> AEXMLElement? {
        children(element, local).first
    }

    /// 深度优先，保持文档顺序。
    static func descendants(_ element: AEXMLElement, _ local: String) -> [AEXMLElement] {
        let target = local.lowercased()
        var result: [AEXMLElement] = []
        func walk(_ node: AEXMLElement) {
            for child in node.children {
                if localName(child.name).lowercased() == target {
                    result.append(child)
                }
                walk(child)
            }
        }
        walk(element)
        return result
    }

    static func attribute(_ element: AEXMLElement, _ names: [String]) -> String? {
        // 先精确匹配（如 r:id），再按本地名匹配。
        for name in names {
            if let value = element.attributes[name] { return value }
        }
        for name in names {
            let target = localName(name).lowercased()
            for (key, value) in element.attributes
            where localName(key).lowercased() == target {
                return value
            }
        }
        return nil
    }

    static func parse(_ data: Data, source: String) throws -> AEXMLDocument {
        do {
            return try AEXMLDocument(xml: data)
        } catch {
            throw AgentError.parse("XML 解析失败（\(source)）：\(error.localizedDescription)")
        }
    }
}

/// EPUB/OPF 内部路径规范化与相对路径解析。
enum RAGArchivePath {
    static func normalize(_ path: String) -> String {
        var components: [String] = []
        for rawComponent in path.split(separator: "/", omittingEmptySubsequences: false) {
            let component = String(rawComponent)
            if component.isEmpty || component == "." { continue }
            if component == ".." {
                if let last = components.last, last != ".." {
                    components.removeLast()
                } else {
                    components.append("..")
                }
            } else {
                components.append(component)
            }
        }
        return components.joined(separator: "/")
    }

    static func stripFragmentAndQuery(_ href: String) -> String {
        var value = href
        if let hash = value.firstIndex(of: "#") {
            value = String(value[..<hash])
        }
        if let query = value.firstIndex(of: "?") {
            value = String(value[..<query])
        }
        return value
    }

    /// 以 `baseFile` 的目录为基准解析 href，并做 URL 解码与 ../ 归一化。
    static func resolve(_ href: String, relativeTo baseFile: String) -> String {
        let cleaned = stripFragmentAndQuery(href).trimmingCharacters(in: .whitespacesAndNewlines)
        let decoded = cleaned.removingPercentEncoding ?? cleaned
        if decoded.hasPrefix("/") {
            return normalize(String(decoded.dropFirst()))
        }
        let baseDirectory = (baseFile as NSString).deletingLastPathComponent
        if baseDirectory.isEmpty || baseDirectory == "." {
            return normalize(decoded)
        }
        return normalize(baseDirectory + "/" + decoded)
    }
}

// MARK: - EPUB 解析器

/// EPUB 解析：ZipFoundation 解包 + AEXML 解析 container/OPF/NCX + SwiftSoup 清理 XHTML。
public final class EPUBParser: DocumentParser {

    public let name = "EPUB"
    public let supportedExtensions = ["epub"]

    /// 扫描页判据（整书预采样与逐章兜底共用）：「平均每张图分摊的有效文字」
    /// 低于该值 → 图是正文扫描页而非插图。扫描页的文字层每图只有几十字符，
    /// 正常插图周围有成千上万字，远超此线。
    static let scannedCharsPerImage = 150.0

    private let providedVLM: VLMService?
    private let config: AgentConfig?
    private lazy var resolvedVLM: VLMService? = {
        guard let config else { return nil }
        // rag_ovis_instances=2 时返回双实例池：实例各自串行、池级并行。
        return LocalModelFactory.shared.makeRAGOCRVLM(config: config)
    }()
    /// OvisOCR3 仅在扫描页且选择 ovis 后端时才解析/加载。
    public var vlm: VLMService? { providedVLM ?? resolvedVLM }

    public init() {
        self.providedVLM = nil
        self.config = nil
    }

    public init(vlm: VLMService?) {
        self.providedVLM = vlm
        self.config = nil
    }

    public init(config: AgentConfig) {
        self.providedVLM = nil
        self.config = config
    }

    private struct ManifestItem {
        let id: String
        let href: String
        let mediaType: String
        let properties: String
        let path: String

        var isXHTML: Bool {
            mediaType.contains("xhtml") || mediaType.contains("html")
                || path.lowercased().hasSuffix(".xhtml")
                || path.lowercased().hasSuffix(".html")
                || path.lowercased().hasSuffix(".htm")
        }
    }

    private struct NavigationEntry {
        let title: String
        let path: String
        let level: Int
    }

    public func parse(url: URL,
                      options: ParserOptions,
                      progress: ((ParserProgress) -> Void)?) async throws -> ParsedDocument {
        let archive = try RAGArchiveReader.open(url)

        guard let containerData = RAGArchiveReader.data(archive, path: "META-INF/container.xml") else {
            throw AgentError.parse("EPUB 缺少 META-INF/container.xml：\(url.lastPathComponent)")
        }
        let container = try RAGXMLHelper.parse(containerData, source: "META-INF/container.xml")
        guard let rootfile = RAGXMLHelper.descendants(container.root, "rootfile").first,
              let declaredOPF = RAGXMLHelper.attribute(rootfile, ["full-path"]),
              !declaredOPF.isEmpty else {
            throw AgentError.parse("EPUB container.xml 未声明 OPF 文件：\(url.lastPathComponent)")
        }

        let opfPath = RAGArchivePath.normalize(declaredOPF.removingPercentEncoding ?? declaredOPF)
        guard let opfData = RAGArchiveReader.data(archive, path: opfPath) else {
            throw AgentError.parse("EPUB 缺少 OPF 文件：\(opfPath)")
        }
        let opf = try RAGXMLHelper.parse(opfData, source: opfPath)

        let title = metadataTitle(opf.root)
        let manifest = parseManifest(opf.root, opfPath: opfPath)
        let spineElement = RAGXMLHelper.firstChild(opf.root, "spine")
        let spinePaths = parseSpinePaths(spineElement, manifest: manifest, archive: archive)

        guard !spinePaths.isEmpty else {
            throw AgentError.parse("EPUB 未找到可读的 spine 章节：\(url.lastPathComponent)")
        }

        let tocID = spineElement.flatMap { RAGXMLHelper.attribute($0, ["toc"]) }
        let (navigation, tocSource) = parseNavigation(archive: archive,
                                                      manifest: manifest,
                                                      tocID: tocID,
                                                      opfPath: opfPath)

        var chapters: [String] = []
        var headings: [OutlineEntry] = []
        var nextPageNumber = 1
        var scannedChapters = 0
        var ocrPages = 0

        // 整书预采样：转制/Kindle 扫描书的文字层常只有目录与点缀（总量可能接近一万），
        // 绝对阈值判不住。采样前 20 个 spine 条目，图片 ≥ 10 张且平均每图
        // 不足 300 字符 → 整书按扫描版处理，含图条目一律走 OCR。
        let scannedBook = Self.looksLikeScannedBook(
            spinePaths: Array(spinePaths.prefix(20)), archive: archive)
        if scannedBook {
            AgentLog.info("EPUB: 预采样判定为扫描版（图片多、平均每图文字不足 300），含图条目将走 OCR")
        }

        for (index, path) in spinePaths.enumerated() {
            guard let data = RAGArchiveReader.data(archive, path: path), !data.isEmpty else {
                continue
            }
            let xhtml = DocumentTextDecoder.decode(data)
            guard let document = try? SwiftSoup.parse(xhtml, path) else { continue }

            let (markdown, documentHeadings) = HTMLMarkdown.convert(document, removeBoilerplate: false)
            let imagePaths = Self.imagePaths(in: document, chapterPath: path)

            var chapter = markdown
            var isScannedChapter = false
            // 纯图片页（扫描版 EPUB）：文字层不足时按图片顺序 OCR，
            // 结果用与 PDF 相同的页标记拼接。
            // 扫描页三层判定（阈值统一为「每图分摊 150 有效文字」）：
            // 层 1 整书：预采样前 20 个 spine 条目，图 ≥10 且平均每图 <150 字
            //   → 转制/Kindle 扫描书（文字层只有目录点缀），所有含图条目一律 OCR；
            // 层 2 逐章：整书未命中时逐章判「文字 < 图数 × 150」——独立扫描章、
            //   常规扫描页（每条目一页一图、文字近乎 0）都在此命中。
            //   单图条目文字 <150 是层 2 的特例（1 × 150），整页插图条目可能被
            //   误 OCR（白识别一次，OCR 空结果时保留原文字层），但保证不漏。
            let effectiveChars = DocumentCleaner.effectiveCharacterCount(markdown)
            if !imagePaths.isEmpty,
               scannedBook
                   || Double(effectiveChars) / Double(imagePaths.count)
                       < Self.scannedCharsPerImage {
                isScannedChapter = true
                let ocr = try await recognizeImages(imagePaths, archive: archive,
                                                    options: options,
                                                    startingPage: nextPageNumber)
                scannedChapters += 1
                ocrPages += ocr.pages
                nextPageNumber += ocr.pages
                if !ocr.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    chapter = ocr.text
                } else if !markdown.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    AgentLog.warning("EPUB: 第 \(index + 1) 章 OCR 无结果，保留原文字层")
                }
            }

            guard !chapter.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }

            let tocTitle = navigation.first(where: { $0.path == path })?.title
                ?? navigation.first(where: {
                    ($0.path as NSString).lastPathComponent == (path as NSString).lastPathComponent
                })?.title
            // 扫描页常常是「每个 spine 项一页」；没有真实 TOC/标题时
            // 不要为每页合成「第 N 节」标题，否则大纲会被几百个条目撑爆。
            let explicitTitle = tocTitle ?? documentHeadings.first?.title
            let chapterTitle = explicitTitle ?? "第 \(index + 1) 节"

            // 章节标题转 Markdown 标题；已有标题时保持原结构。
            if MarkdownOutline.scan(chapter).isEmpty,
               !isScannedChapter || explicitTitle != nil {
                chapter = "## \(chapterTitle)\n\n" + chapter
            }
            chapters.append(chapter)

            let scanned = MarkdownOutline.scan(chapter)
            if scanned.isEmpty {
                if !isScannedChapter || explicitTitle != nil {
                    headings.append(OutlineEntry(level: 2, title: chapterTitle))
                }
            } else {
                headings.append(contentsOf: scanned)
            }
        }

        guard !chapters.isEmpty else {
            throw AgentError.parse("EPUB 未解析出正文：\(url.lastPathComponent)")
        }

        // 大纲取条目多的一方：正文 headings 反映实际排版结构（含伪标题识别），
        // 但很多书只有 NCX/nav 目录才完整；反之纯排版书 headings 更细。
        var outline = MarkdownOutline.build(headings)
        if headings.count < navigation.count {
            outline = MarkdownOutline.build(navigation.map {
                OutlineEntry(level: $0.level, title: $0.title)
            })
        }

        var metadata: [String: String] = [
            "engine": "epub",
            "chapters": String(chapters.count),
            "toc": tocSource,
            "opf": opfPath,
        ]
        if scannedChapters > 0 {
            metadata["ocr_engine"] = options.epubOCREngine.rawValue
            metadata["ocr_pages"] = String(ocrPages)
            metadata["scanned_chapters"] = String(scannedChapters)
        }
        if !title.isEmpty { metadata["title"] = title }

        var parsed = ParsedDocument(text: chapters.joined(separator: "\n\n"),
                                    source: url.lastPathComponent,
                                    outline: outline,
                                    metadata: metadata)
        parsed = finish(parsed, options: options)
        return parsed
    }

    // MARK: - OPF

    private func metadataTitle(_ root: AEXMLElement) -> String {
        let metadata = RAGXMLHelper.firstChild(root, "metadata") ?? root
        for element in RAGXMLHelper.descendants(metadata, "title") {
            let value = element.value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if !value.isEmpty { return value }
        }
        return ""
    }

    private func parseManifest(_ root: AEXMLElement, opfPath: String) -> [ManifestItem] {
        let manifestElement = RAGXMLHelper.firstChild(root, "manifest") ?? root
        return RAGXMLHelper.children(manifestElement, "item").compactMap { item in
            guard let id = RAGXMLHelper.attribute(item, ["id"]),
                  let href = RAGXMLHelper.attribute(item, ["href"]) else { return nil }
            return ManifestItem(
                id: id,
                href: href,
                mediaType: (RAGXMLHelper.attribute(item, ["media-type"]) ?? "").lowercased(),
                properties: (RAGXMLHelper.attribute(item, ["properties"]) ?? "").lowercased(),
                path: RAGArchivePath.resolve(href, relativeTo: opfPath))
        }
    }

    /// spine 缺失时回退到 manifest 中的 XHTML 项，再回退到归档内全部 HTML/XHTML。
    private func parseSpinePaths(_ spine: AEXMLElement?,
                                 manifest: [ManifestItem],
                                 archive: Archive) -> [String] {
        var paths: [String] = []
        var seen = Set<String>()

        if let spine {
            for itemref in RAGXMLHelper.children(spine, "itemref") {
                guard let idref = RAGXMLHelper.attribute(itemref, ["idref"]),
                      let item = manifest.first(where: { $0.id == idref }),
                      seen.insert(item.path).inserted else { continue }
                paths.append(item.path)
            }
        }

        if paths.isEmpty {
            for item in manifest where item.isXHTML {
                if seen.insert(item.path).inserted {
                    paths.append(item.path)
                }
            }
        }

        if paths.isEmpty {
            let htmlExtensions = ["xhtml", "html", "htm"]
            let candidates = RAGArchiveReader.filePaths(archive)
                .filter { htmlExtensions.contains(($0 as NSString).pathExtension.lowercased()) }
                .sorted()
            for path in candidates {
                if seen.insert(path).inserted {
                    paths.append(path)
                }
            }
        }
        return paths
    }

    // MARK: - 目录（NCX / nav.xhtml）

    private func parseNavigation(archive: Archive,
                                 manifest: [ManifestItem],
                                 tocID: String?,
                                 opfPath: String) -> ([NavigationEntry], String) {
        let ncxItem = manifest.first(where: { $0.id == tocID })
            ?? manifest.first(where: { $0.mediaType == "application/x-dtbncx+xml" })
            ?? manifest.first(where: { $0.path.lowercased().hasSuffix(".ncx") })

        if let ncxItem,
           let data = RAGArchiveReader.data(archive, path: ncxItem.path),
           let document = try? AEXMLDocument(xml: data) {
            let entries = parseNCX(document, ncxPath: ncxItem.path)
            if !entries.isEmpty { return (entries, "ncx") }
        }

        let navItem = manifest.first(where: { $0.properties.split(separator: " ").contains("nav") })
            ?? manifest.first(where: {
                let name = ($0.path as NSString).lastPathComponent.lowercased()
                return name == "nav.xhtml" || name == "toc.xhtml"
            })

        if let navItem,
           let data = RAGArchiveReader.data(archive, path: navItem.path) {
            let entries = parseNav(data, navPath: navItem.path)
            if !entries.isEmpty { return (entries, "nav") }
        }

        return ([], "none")
    }

    private func parseNCX(_ document: AEXMLDocument, ncxPath: String) -> [NavigationEntry] {
        guard let navMap = RAGXMLHelper.descendants(document.root, "navMap").first else { return [] }
        var entries: [NavigationEntry] = []

        func walk(_ parent: AEXMLElement, level: Int) {
            for point in RAGXMLHelper.children(parent, "navPoint") {
                let labelElement = RAGXMLHelper.firstChild(point, "navLabel")
                    .flatMap { RAGXMLHelper.firstChild($0, "text") }
                let title = (labelElement?.value ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let src = RAGXMLHelper.firstChild(point, "content")
                    .flatMap { RAGXMLHelper.attribute($0, ["src"]) }
                if !title.isEmpty, let src {
                    entries.append(NavigationEntry(
                        title: title,
                        path: RAGArchivePath.resolve(src, relativeTo: ncxPath),
                        level: level))
                }
                walk(point, level: level + 1)
            }
        }
        walk(navMap, level: 1)
        return entries
    }

    private func parseNav(_ data: Data, navPath: String) -> [NavigationEntry] {
        let html = DocumentTextDecoder.decode(data)
        guard let document = try? SwiftSoup.parse(html, navPath) else { return [] }

        let navElements = (try? document.select("nav").array()) ?? []
        let container: Element?
        if let tocNav = navElements.first(where: {
            (try? $0.attr("epub:type")) == "toc" || (try? $0.attr("type")) == "toc"
        }) {
            container = tocNav
        } else if let firstNav = navElements.first {
            container = firstNav
        } else {
            container = document.body()
        }
        guard let container, let list = try? container.select("ol").first() else { return [] }

        var entries: [NavigationEntry] = []
        func walk(_ listElement: Element, level: Int) {
            for item in listElement.children() where item.tagName().lowercased() == "li" {
                if let anchor = item.children().first(where: {
                    $0.tagName().lowercased() == "a"
                }) {
                    let title = ((try? anchor.text()) ?? "")
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    let href = (try? anchor.attr("href")) ?? ""
                    if !title.isEmpty, !href.isEmpty {
                        entries.append(NavigationEntry(
                            title: title,
                            path: RAGArchivePath.resolve(href, relativeTo: navPath),
                            level: level))
                    }
                }
                for child in item.children() where child.tagName().lowercased() == "ol" {
                    walk(child, level: level + 1)
                }
            }
        }
        walk(list, level: 1)
        return entries
    }

    // MARK: - 扫描页 OCR

    /// 预采样前 N 个 spine 条目判定整书是否扫描版：
    /// 图片 ≥ 10 张且「总有效文字 / 图片数」不足 150 字符 → 扫描版
    /// （与逐章判定共用 scannedCharsPerImage 判据）。
    /// 扫描书的文字层（目录/页眉/内置 OCR 层）每图只有几十字符，
    /// 总量可能接近一万；正常图文书每张插图周围有成千上万字，远超此线。
    static func looksLikeScannedBook(spinePaths: [String], archive: Archive) -> Bool {
        guard !spinePaths.isEmpty else { return false }
        var totalImages = 0
        var totalChars = 0
        for path in spinePaths {
            guard let data = RAGArchiveReader.data(archive, path: path), !data.isEmpty,
                  let document = try? SwiftSoup.parse(
                      DocumentTextDecoder.decode(data), path) else { continue }
            let images = imagePaths(in: document, chapterPath: path).count
            guard images > 0 else { continue }
            totalImages += images
            totalChars += DocumentCleaner.effectiveCharacterCount(
                HTMLMarkdown.convert(document, removeBoilerplate: false).markdown)
        }
        guard totalImages >= 10 else { return false }
        return Double(totalChars) / Double(totalImages) < 150
    }

    /// 按文档顺序收集 XHTML 中的图片路径（兼容 `<img src>` 与 SVG `<image xlink:href>`）。
    private static func imagePaths(in document: Document, chapterPath: String) -> [String] {
        var result: [String] = []
        var seen = Set<String>()

        func walk(_ node: Node) {
            if let element = node as? Element {
                let tag = element.tagName().lowercased()
                if tag == "img" || tag == "image" {
                    var raw = (try? element.attr("src")) ?? ""
                    if raw.isEmpty { raw = (try? element.attr("xlink:href")) ?? "" }
                    if raw.isEmpty { raw = (try? element.attr("href")) ?? "" }
                    raw = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !raw.isEmpty, !raw.hasPrefix("data:") {
                        let path = RAGArchivePath.resolve(raw, relativeTo: chapterPath)
                        if !path.isEmpty, seen.insert(path).inserted {
                            result.append(path)
                        }
                    }
                }
            }
            for child in node.getChildNodes() { walk(child) }
        }
        walk(document)
        return result
    }

    /// 对一章内的图片逐页 OCR；返回带页标记的 Markdown 与页数。
    ///
    /// 两条后端路径都先做统一归一化（缩放/灰度/对比度/锐化/可选二值化）：
    /// - `.vision`：每批最多 8 页并发调用 Apple Vision，全局受闸门限制；
    /// - `.ovis` / `.custom`：借 OCR VLM 实例池并发识别（多实例 2 路，单实例池内仍串行）。
    private func recognizeImages(_ paths: [String],
                                 archive: Archive,
                                 options: ParserOptions,
                                 startingPage: Int)
        async throws -> (text: String, pages: Int, recognized: Int) {
        guard !paths.isEmpty else { return ("", 0, 0) }

        let ovis: VLMService?
        if options.epubOCREngine == .ovis || options.epubOCREngine == .custom {
            guard let service = vlm, service.isAvailable else {
                throw AgentError.parse(
                    "扫描版 EPUB 需要 OCR 视觉模型，但视觉模型不可用（未配置本地 VLM）")
            }
            ovis = service
        } else {
            ovis = nil
        }

        let batchSize = PDFPageRenderer.maxConcurrentPages
        var sections: [String] = []
        var recognized = 0
        var page = startingPage
        var cursor = 0

        while cursor < paths.count {
            if Task.isCancelled { throw AgentError.cancelled }
            let end = min(cursor + batchSize, paths.count)
            let batch = Array(paths[cursor..<end])

            // 图片解码按批串行执行，避免多个线程同时读同一个 ZIP 句柄。
            var images: [PDFPageImage] = []
            images.reserveCapacity(batch.count)
            for (offset, relativePath) in batch.enumerated() {
                guard let data = RAGArchiveReader.data(archive, path: relativePath),
                      let image = Self.decodeImage(data) else {
                    AgentLog.warning("EPUB: 无法解码图片 \(relativePath)，已跳过")
                    continue
                }
                images.append(PDFPageImage(pageIndex: page + offset, image: image))
            }

            if !images.isEmpty {
                let texts: [(Int, String)]
                if let ovis {
                    // 归一化在批内串行做（CIImage 处理占 CPU），
                    // 识别借实例池并发；结果收集后汇总，避免竞态。
                    let normalized: [(Int, CGImage)] = images.map {
                        ($0.pageIndex, Self.normalizedImage($0.image, options: options))
                    }
                    texts = try await withThrowingTaskGroup(
                        of: (Int, String).self) { group in
                        for (pageIndex, source) in normalized {
                            group.addTask {
                                do {
                                    try Task.checkCancellation()
                                    let raw = try await ovis.recognize(
                                        image: source,
                                        prompt: OvisPrompt.pageToMarkdown,
                                        maxTokens: OvisPDFParser.maxTokens)
                                    return (pageIndex, OvisPDFParser.cleanVLMOutput(raw))
                                } catch is CancellationError {
                                    throw CancellationError()
                                } catch {
                                    // 单页失败跳过，不拖垮整批。
                                    AgentLog.warning(
                                        "EPUB: 第 \(pageIndex) 页 Ovis OCR 失败，已跳过: "
                                        + error.localizedDescription)
                                    return (pageIndex, "")
                                }
                            }
                        }
                        var results: [(Int, String)] = []
                        results.reserveCapacity(normalized.count)
                        for try await item in group {
                            results.append(item)
                        }
                        return results
                    }
                } else {
                    texts = try await visionBatch(images, options: options)
                }

                for (pageNumber, text) in texts.sorted(by: { $0.0 < $1.0 }) {
                    sections.append(DocumentCleaner.pageMarker(pageNumber))
                    let body = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !body.isEmpty {
                        sections.append(body)
                        recognized += 1
                    }
                }
            }

            page += batch.count
            cursor = end
        }

        return (sections.joined(separator: "\n\n"),
                page - startingPage, recognized)
    }

    private func visionBatch(_ images: [PDFPageImage],
                             options: ParserOptions) async throws -> [(Int, String)] {
        // 复用 PDF 的 Vision 批处理：统一归一化、全局并发闸门与取消传播。
        try await VisionPDFParser.recognizeBatch(
            images,
            preprocess: options.ocrPreprocess,
            binarize: options.ocrBinarize,
            languages: options.ocrLanguages)
    }

    private static func normalizedImage(_ image: CGImage,
                                        options: ParserOptions) -> CGImage {
        guard options.ocrPreprocess else { return image }
        return OCRImagePreprocessor.normalize(image, binarize: options.ocrBinarize)
            ?? image
    }

    private static func decodeImage(_ data: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            return nil
        }
        return CGImageSourceCreateImageAtIndex(
            source, 0,
            [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
    }
}

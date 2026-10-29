import Foundation
import CoreGraphics
import PDFKit

/// 单页文本层提取结果。
struct PDFPageTextExtraction {
    let text: String
    /// 页面内图片 XObject 数量（含 Form XObject 递归，封顶 256）。
    let imageObjectCount: Int
    /// true 表示该页跳过了 PDFKit，走 `CGPDFScanner` 有界提取。
    let usedScanner: Bool
}

/// 有界 PDF 文本层提取。
///
/// PDFKit 的 `PDFPage.string` 最终走 `CGPDFPageCopyString` →
/// `PageLayout::getOrderedLayout` → `CRCTLD::findWhitespaceRegions`。
/// 页面里的图片 XObject 会让这套文本行布局的病态空白区域分析持续膨胀：
/// 实测 383 页 319MB 的 PDF 中，第 18 页（392 张图）单页物理内存涨到 42GB
/// 且长时间不返回。
///
/// 因此不能只按图片数量判断：扫描书常见「每页只有 1 张 2000×3000 的全页图」，
/// 图片少但像素极大，同样可能触发 PDFKit 布局爆炸。这里的策略：
/// - 图片数量 ≥ `imageHeavyThreshold`、单图 ≥ `riskySingleImagePixels`、
///   或页面图片总像素 ≥ `riskyTotalImagePixels` 的页面不再调用 PDFKit，
///   改用 `CGPDFScanner` 逐操作符读取文本（字体 + ToUnicode + 文本状态），
///   不触发布局与障碍物分析，页级内存与图片数量/像素数无关；
/// - 低图片页继续走 `PDFPage.string`，保持原有成熟的阅读顺序与分段行为；
/// - 扫描器输出为空时由解析器自然进入 OCR 路径，绝不回退到 PDFKit 文本层。
enum PDFTextExtractor {

    /// 图片 XObject 达到该数量后不再调用 PDFKit 文本层。
    static let imageHeavyThreshold = 16
    /// 单张图片像素数达到该值时不再调用 PDFKit（覆盖每页一张全页扫描图）。
    static let riskySingleImagePixels: Int64 = 2_000_000
    /// 页面图片总像素达到该值时不再调用 PDFKit（覆盖多张中等图片叠加）。
    static let riskyTotalImagePixels: Int64 = 6_000_000
    /// 跨文件同时进行的文本层提取上限。多文件并发时每个文件仍有 2-4 个
    /// worker，若不设全局闸门，6 个文件会同时有 20 多页进入 PDFKit 布局；
    /// 一旦某页触发病态分析，内存会成倍叠加。文本提取远快于 OCR，
    /// 固定 3 路足以维持吞吐，又能把峰值乘数压住。
    static var requestedConcurrency: Int?
    static var maxConcurrentExtractions: Int { requestedConcurrency ?? 3 }
    private static let extractionGate = AsyncSemaphore(value: maxConcurrentExtractions)

    struct ImageProfile: Equatable {
        let count: Int
        let pixelCount: Int64
        let largestPixelCount: Int64

        var isMemoryRisky: Bool {
            count >= PDFTextExtractor.imageHeavyThreshold
                || largestPixelCount >= PDFTextExtractor.riskySingleImagePixels
                || pixelCount >= PDFTextExtractor.riskyTotalImagePixels
        }
    }
    /// 单页输出上限：防御异常 PDF 无限输出文本。
    static let maxCharactersPerPage = 1_000_000
    /// Form XObject 递归深度上限（同时防御自引用循环）。
    static let maxFormDepth = 8
    static let maxFormScans = 512
    private static let maxResourcesDepth = 32
    static let maxCountedImages = 256
    static let maxCMapBytes = 4 * 1024 * 1024
    static let maxCMapEntries = 65_536
    static let maxCMapRangeSpan: UInt32 = 65_536
    static let maxCMapTokens = 200_000
    static let maxCMapBlocks = 1_024
    static let maxDecoders = 64
    static let maxStateDepth = 64

    /// 提取单页文本层。调用方应在 autoreleasepool 中调用。
    static func text(for page: PDFPage) -> PDFPageTextExtraction {
        guard let cgPage = page.pageRef else {
            return PDFPageTextExtraction(text: page.string ?? "",
                                         imageObjectCount: 0,
                                         usedScanner: false)
        }
        let profile = imageProfile(for: cgPage)
        guard !profile.isMemoryRisky else {
            return PDFPageTextExtraction(text: scanText(for: cgPage),
                                         imageObjectCount: profile.count,
                                         usedScanner: true)
        }
        return PDFPageTextExtraction(text: page.string ?? "",
                                     imageObjectCount: profile.count,
                                     usedScanner: false)
    }

    /// 跨文件限流的文本层提取：页面对象创建、PDFKit 布局与 autorelease
    /// 都在闸门内完成，避免多文件并发时同时进入病态页面。
    ///
    /// PDFPage 由 PDFDocument 惰性创建；必须每页在池内创建并释放，
    /// 否则整本书的页对象会一直挂在线程的自动释放池里。
    static func extract(for document: PDFDocument,
                        pageIndex: Int) async throws -> PDFPageTextExtraction? {
        guard await extractionGate.acquire() else { throw AgentError.cancelled }
        defer { extractionGate.release() }
        guard !Task.isCancelled else { throw AgentError.cancelled }

        let beforeMB = ProcessMemory.footprintMB()
        let extraction: PDFPageTextExtraction? = autoreleasepool {
            guard let page = document.page(at: pageIndex) else { return nil }
            return text(for: page)
        }
        let afterMB = ProcessMemory.footprintMB()
        // 单页文本提取正常只有几十 MB 波动；一旦某页出现 42GB 级病态布局，
        // 这条日志会直接指出页码，避免再次靠猜定位。
        // 只看单页增量：进程本来已占用较高（如 MLX 权重）时不应逐页刷屏。
        if afterMB - beforeMB >= 256 {
            let route = extraction?.usedScanner == true ? "scanner" : "PDFKit"
            AgentLog.warning(
                "[pdf] 第 \(pageIndex + 1) 页文本提取内存异常："
                    + "+\(Int(afterMB - beforeMB))MB，当前 \(Int(afterMB))MB"
                    + "（\(route)，图片 \(extraction?.imageObjectCount ?? 0) 个）")
        }
        return extraction
    }

    /// 统计页面资源里的图片 XObject 数量（含 Form XObject 内的图片）。
    static func imageObjectCount(for page: CGPDFPage) -> Int {
        imageProfile(for: page).count
    }

    /// 页面图片画像：数量 + 像素总量，用于判断 PDFKit 文本布局是否安全。
    static func imageProfile(for page: CGPDFPage) -> ImageProfile {
        guard let pageDictionary = page.dictionary,
              let resources = resources(in: pageDictionary) else {
            return ImageProfile(count: 0, pixelCount: 0, largestPixelCount: 0)
        }
        let counter = ImageObjectCounter()
        counter.scan(resources: resources, depth: 0)
        return ImageProfile(count: counter.total,
                            pixelCount: counter.pixelCount,
                            largestPixelCount: counter.largestPixelCount)
    }

    /// 用 `CGPDFScanner` 提取单页文本，不依赖页面布局。
    static func scanText(for page: CGPDFPage) -> String {
        let content = CGPDFContentStreamCreateWithPage(page)
        let table = makeOperatorTable()
        let state = PDFTextScanState(page: page, table: table)
        let scanner = CGPDFScannerCreate(content, table,
                                         Unmanaged.passUnretained(state).toOpaque())
        _ = CGPDFScannerScan(scanner)
        return state.output
    }

    /// 页面字典可能从 `/Parent` 继承 `/Resources`。
    static func resources(in pageDictionary: CGPDFDictionaryRef) -> CGPDFDictionaryRef? {
        var current: CGPDFDictionaryRef? = pageDictionary
        var depth = 0
        while let dictionary = current, depth < maxResourcesDepth {
            var value: CGPDFDictionaryRef?
            if CGPDFDictionaryGetDictionary(dictionary, "Resources", &value), let value {
                return value
            }
            var parent: CGPDFDictionaryRef?
            if CGPDFDictionaryGetDictionary(dictionary, "Parent", &parent), let parent {
                current = parent
            } else {
                return nil
            }
            depth += 1
        }
        return nil
    }

    private static func makeOperatorTable() -> CGPDFOperatorTableRef {
        let table = CGPDFOperatorTableCreate()!
        let callbacks: [(String, CGPDFOperatorCallback)] = [
            ("BT", btCallback), ("ET", etCallback),
            ("Tf", tfCallback), ("TL", tlCallback),
            ("Td", tdCallback), ("TD", tdSetLeadingCallback),
            ("Tm", tmCallback), ("T*", tstarCallback),
            ("Tj", tjCallback), ("'", quoteCallback), ("\"", doubleQuoteCallback),
            ("TJ", tjArrayCallback),
            ("Tc", tcCallback), ("Tw", twCallback),
            ("q", qCallback), ("Q", qPopCallback),
            ("Do", doCallback),
        ]
        for (name, callback) in callbacks {
            CGPDFOperatorTableSetCallback(table, name, callback)
        }
        return table
    }
}

// MARK: - 图片 XObject 计数

private final class ImageObjectCounter {
    var total = 0
    /// 页面图片声明的像素总数（Width × Height 累加，含 Form XObject 内图片）。
    var pixelCount: Int64 = 0
    /// 单张图片的最大像素数，用于识别「每页一张全页大图」的扫描书。
    var largestPixelCount: Int64 = 0
    private var visitedForms: Set<CGPDFStreamRef> = []

    func scan(resources: CGPDFDictionaryRef, depth: Int) {
        var xobjects: CGPDFDictionaryRef?
        guard CGPDFDictionaryGetDictionary(resources, "XObject", &xobjects),
              let xobjects else { return }
        CGPDFDictionaryApplyBlock(xobjects, { key, _, _ in
            guard self.total < PDFTextExtractor.maxCountedImages else { return false }
            var stream: CGPDFStreamRef?
            guard CGPDFDictionaryGetStream(xobjects, key, &stream), let stream,
                  let dictionary = CGPDFStreamGetDictionary(stream) else { return true }
            var subtypePointer: UnsafePointer<CChar>?
            guard CGPDFDictionaryGetName(dictionary, "Subtype", &subtypePointer),
                  let subtypePointer else { return true }
            switch String(cString: subtypePointer) {
            case "Image":
                self.total += 1
                let pixels = Self.pixelCount(of: dictionary)
                self.pixelCount += pixels
                self.largestPixelCount = max(self.largestPixelCount, pixels)
            case "Form" where depth < PDFTextExtractor.maxFormDepth:
                guard self.visitedForms.insert(stream).inserted else { return true }
                var formResources: CGPDFDictionaryRef?
                let child = (CGPDFDictionaryGetDictionary(dictionary, "Resources", &formResources)
                    ? formResources : nil) ?? resources
                self.scan(resources: child, depth: depth + 1)
            default:
                break
            }
            return true
        }, nil)
    }

    /// 只读声明的 `/Width`、`/Height`，不解码图片数据。
    /// 尺寸缺失或异常时按 0 计，仍由图片数量阈值兜底。
    private static func pixelCount(of dictionary: CGPDFDictionaryRef) -> Int64 {
        var width: CGPDFInteger = 0
        var height: CGPDFInteger = 0
        guard CGPDFDictionaryGetInteger(dictionary, "Width", &width),
              CGPDFDictionaryGetInteger(dictionary, "Height", &height),
              width > 0, height > 0 else { return 0 }
        let (pixels, overflow) = Int64(width).multipliedReportingOverflow(by: Int64(height))
        return overflow ? Int64.max : pixels
    }
}

// MARK: - ToUnicode CMap

private enum PDFCMapToken {
    case hex([UInt8])
    case arrayStart
    case arrayEnd
    case other
}

private struct PDFCMap {
    var map: [UInt32: String]
    var codeLength: Int
}

private enum PDFCMapParser {

    static func parse(_ data: Data) -> PDFCMap? {
        guard data.count <= PDFTextExtractor.maxCMapBytes,
              let text = String(data: data, encoding: .isoLatin1) else { return nil }
        var map: [UInt32: String] = [:]
        var codeLength = 0

        if let begin = text.range(of: "begincodespacerange"),
           let end = text.range(of: "endcodespacerange",
                                range: begin.upperBound..<text.endIndex) {
            for token in tokenize(text[begin.upperBound..<end.lowerBound]) {
                if case .hex(let bytes) = token {
                    codeLength = max(codeLength, bytes.count)
                }
            }
        }

        var searchStart = text.startIndex
        var blocks = 0
        while blocks < PDFTextExtractor.maxCMapBlocks,
              let begin = text.range(of: "beginbfchar", range: searchStart..<text.endIndex),
              let end = text.range(of: "endbfchar",
                                   range: begin.upperBound..<text.endIndex) {
            let tokens = tokenize(text[begin.upperBound..<end.lowerBound])
            var index = 0
            while index + 1 < tokens.count {
                if case .hex(let source) = tokens[index],
                   case .hex(let destination) = tokens[index + 1],
                   map.count < PDFTextExtractor.maxCMapEntries {
                    let value = unicodeString(destination)
                    if !value.isEmpty {
                        map[hexValue(source)] = value
                    }
                }
                index += 2
            }
            searchStart = end.upperBound
            blocks += 1
        }

        searchStart = text.startIndex
        blocks = 0
        while blocks < PDFTextExtractor.maxCMapBlocks,
              let begin = text.range(of: "beginbfrange", range: searchStart..<text.endIndex),
              let end = text.range(of: "endbfrange",
                                   range: begin.upperBound..<text.endIndex) {
            let tokens = tokenize(text[begin.upperBound..<end.lowerBound])
            var index = 0
            while index + 2 < tokens.count {
                guard case .hex(let lowBytes) = tokens[index],
                      case .hex(let highBytes) = tokens[index + 1] else {
                    index += 1
                    continue
                }
                let low = hexValue(lowBytes)
                let high = hexValue(highBytes)
                guard low <= high, high - low < PDFTextExtractor.maxCMapRangeSpan else {
                    index += 1
                    continue
                }

                switch tokens[index + 2] {
                case .hex(let destination):
                    let base = hexValue(destination)
                    let byteCount = max(1, min(destination.count, 4))
                    var offset: UInt32 = 0
                    while offset <= high - low,
                          map.count < PDFTextExtractor.maxCMapEntries {
                        let value = unicodeString(integerBytes(base &+ offset, count: byteCount))
                        if !value.isEmpty {
                            map[low + offset] = value
                        }
                        offset += 1
                    }
                    index += 3
                case .arrayStart:
                    var destinationIndex = index + 3
                    var code = low
                    while destinationIndex < tokens.count, code <= high {
                        if case .arrayEnd = tokens[destinationIndex] { break }
                        if case .hex(let destination) = tokens[destinationIndex],
                           map.count < PDFTextExtractor.maxCMapEntries {
                            let value = unicodeString(destination)
                            if !value.isEmpty {
                                map[code] = value
                            }
                            code += 1
                        }
                        destinationIndex += 1
                    }
                    index = max(destinationIndex + 1, index + 3)
                default:
                    index += 1
                }
            }
            searchStart = end.upperBound
            blocks += 1
        }

        guard !map.isEmpty else { return nil }
        return PDFCMap(map: map, codeLength: max(1, codeLength))
    }

    // MARK: 词法

    private static func tokenize(_ text: Substring) -> [PDFCMapToken] {
        var tokens: [PDFCMapToken] = []
        var index = text.startIndex
        while index < text.endIndex {
            if tokens.count >= PDFTextExtractor.maxCMapTokens { break }
            let character = text[index]
            if character == "<" {
                guard let close = text[index...].firstIndex(of: ">") else { break }
                let hexStart = text.index(after: index)
                var hex = String(text[hexStart..<close])
                hex.removeAll { $0.isWhitespace }
                if hex.count % 2 == 0, let bytes = hexBytes(hex) {
                    tokens.append(.hex(bytes))
                }
                index = text.index(after: close)
            } else if character == "[" {
                tokens.append(.arrayStart)
                index = text.index(after: index)
            } else if character == "]" {
                tokens.append(.arrayEnd)
                index = text.index(after: index)
            } else if character.isWhitespace {
                index = text.index(after: index)
            } else {
                var end = index
                while end < text.endIndex,
                      !text[end].isWhitespace,
                      text[end] != "<", text[end] != "[", text[end] != "]" {
                    end = text.index(after: end)
                }
                tokens.append(.other)
                index = end
            }
        }
        return tokens
    }

    private static func hexBytes(_ hex: String) -> [UInt8]? {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(hex.count / 2)
        var index = hex.startIndex
        while index < hex.endIndex {
            let next = hex.index(index, offsetBy: 2)
            guard let byte = UInt8(hex[index..<next], radix: 16) else { return nil }
            bytes.append(byte)
            index = next
        }
        return bytes
    }

    private static func hexValue(_ bytes: [UInt8]) -> UInt32 {
        var value: UInt32 = 0
        for byte in bytes.prefix(4) {
            value = (value << 8) | UInt32(byte)
        }
        return value
    }

    private static func integerBytes(_ value: UInt32, count: Int) -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: count)
        var remaining = value
        var index = count - 1
        while index >= 0 {
            bytes[index] = UInt8(remaining & 0xFF)
            remaining >>= 8
            index -= 1
        }
        return bytes
    }

    /// ToUnicode 的目标串是 UTF-16BE；去掉 NUL/替换符/控制符，避免污染正文。
    private static func unicodeString(_ bytes: [UInt8]) -> String {
        guard !bytes.isEmpty else { return "" }
        var units: [UInt16] = []
        units.reserveCapacity((bytes.count + 1) / 2)
        var index = 0
        while index + 1 < bytes.count {
            units.append(UInt16(bytes[index]) << 8 | UInt16(bytes[index + 1]))
            index += 2
        }
        if index < bytes.count {
            units.append(UInt16(bytes[index]))
        }
        var text = String(decoding: units, as: UTF16.self)
        text.removeAll { character in
            for scalar in character.unicodeScalars {
                if scalar.value == 0 || scalar.value == 0xFFFD { return true }
                if scalar.value < 0x20, scalar.value != 0x09 { return true }
            }
            return false
        }
        return text
    }
}

// MARK: - 字体解码

private final class PDFFontDecoder {
    let codeLength: Int
    let map: [UInt32: String]
    let allowsWinAnsiFallback: Bool

    init(codeLength: Int, map: [UInt32: String], allowsWinAnsiFallback: Bool) {
        self.codeLength = max(1, min(2, codeLength))
        self.map = map
        self.allowsWinAnsiFallback = allowsWinAnsiFallback
    }

    static func make(fontDictionary: CGPDFDictionaryRef) -> PDFFontDecoder {
        var subtypePointer: UnsafePointer<CChar>?
        let subtype: String
        if CGPDFDictionaryGetName(fontDictionary, "Subtype", &subtypePointer),
           let subtypePointer {
            subtype = String(cString: subtypePointer)
        } else {
            subtype = ""
        }
        let isType0 = subtype == "Type0"

        var data: Data?
        var toUnicode: CGPDFStreamRef?
        if CGPDFDictionaryGetStream(fontDictionary, "ToUnicode", &toUnicode),
           let toUnicode {
            var format = CGPDFDataFormat.raw
            if let cfData = CGPDFStreamCopyData(toUnicode, &format) {
                let raw = cfData as Data
                if raw.count <= PDFTextExtractor.maxCMapBytes {
                    data = raw
                }
            }
        }
        let cmap = data.flatMap(PDFCMapParser.parse)
        // 简单字体（除 Type0 外）的字符串一律按单字节字符码解释，
        // 即使其 ToUnicode 的 codespace 写成了 2 字节；否则 WinAnsi
        // 文本会被两两拼接，出现大量丢字。
        let codeLength = isType0 ? (cmap?.codeLength ?? 2) : 1
        return PDFFontDecoder(codeLength: codeLength,
                              map: cmap?.map ?? [:],
                              allowsWinAnsiFallback: !isType0)
    }

    func decode(_ bytes: [UInt8]) -> String {
        guard !bytes.isEmpty else { return "" }
        var result = ""
        if codeLength == 2 {
            var index = 0
            while index + 1 < bytes.count {
                let code = UInt32(bytes[index]) << 8 | UInt32(bytes[index + 1])
                if let text = map[code] { result += text }
                index += 2
            }
            if index < bytes.count, let text = map[UInt32(bytes[index])] {
                result += text
            }
        } else {
            for byte in bytes {
                if let text = map[UInt32(byte)] {
                    result += text
                } else if allowsWinAnsiFallback, let text = Self.winAnsi(byte) {
                    result += text
                }
            }
        }
        return result
    }

    private static func winAnsi(_ byte: UInt8) -> String? {
        if byte < 0x20 { return byte == 0x09 ? " " : nil }
        if byte < 0x7F { return String(UnicodeScalar(byte)) }
        if byte == 0x7F { return nil }
        return String(bytes: [byte], encoding: .windowsCP1252)
    }
}

// MARK: - 文本状态机

private final class PDFTextScanState {
    let page: CGPDFPage
    let operatorTable: CGPDFOperatorTableRef

    var resources: CGPDFDictionaryRef?
    var fontDictionary: CGPDFDictionaryRef?
    var decoder: PDFFontDecoder?
    var fontSize: CGFloat = 12
    var leading: CGFloat = 0
    var wordSpacing: CGFloat = 0
    var characterSpacing: CGFloat = 0

    var lineX: CGFloat = 0
    var lineY: CGFloat = 0
    var cursorX: CGFloat = 0
    var cursorY: CGFloat = 0
    var lastRunEndX: CGFloat?
    var lastRunY: CGFloat?
    var lastRunText = ""

    var output = ""
    private var outputCount = 0
    private var stateStack: [SavedState] = []
    private var decoderCache: [CGPDFDictionaryRef: PDFFontDecoder] = [:]
    private var formDepth = 0
    private var formScans = 0

    private struct SavedState {
        var decoder: PDFFontDecoder?
        var fontSize: CGFloat
        var leading: CGFloat
        var wordSpacing: CGFloat
        var characterSpacing: CGFloat
        var lineX: CGFloat
        var lineY: CGFloat
        var cursorX: CGFloat
        var cursorY: CGFloat
        var lastRunEndX: CGFloat?
        var lastRunY: CGFloat?
        var lastRunText: String
    }

    init(page: CGPDFPage, table: CGPDFOperatorTableRef) {
        self.page = page
        self.operatorTable = table
        guard let dictionary = page.dictionary,
              let resources = PDFTextExtractor.resources(in: dictionary) else { return }
        self.resources = resources
        var fonts: CGPDFDictionaryRef?
        if CGPDFDictionaryGetDictionary(resources, "Font", &fonts) {
            fontDictionary = fonts
        }
    }

    // MARK: 字体

    func setFont(name: String) {
        guard let fontDictionary else {
            decoder = nil
            return
        }
        var reference: CGPDFDictionaryRef?
        guard CGPDFDictionaryGetDictionary(fontDictionary, name, &reference),
              let font = reference else {
            decoder = nil
            return
        }
        if let cached = decoderCache[font] {
            decoder = cached
            return
        }
        let created = PDFFontDecoder.make(fontDictionary: font)
        if decoderCache.count < PDFTextExtractor.maxDecoders {
            decoderCache[font] = created
        }
        decoder = created
    }

    // MARK: 文本对象 / 定位

    func beginText() {
        lineX = 0
        lineY = 0
        cursorX = 0
        cursorY = 0
    }

    func setLineOffset(tx: CGFloat, ty: CGFloat) {
        lineX += tx
        lineY += ty
        cursorX = lineX
        cursorY = lineY
    }

    func setTextMatrix(e: CGFloat, f: CGFloat) {
        lineX = e
        lineY = f
        cursorX = e
        cursorY = f
    }

    func nextLine() {
        lineY -= leading
        cursorX = lineX
        cursorY = lineY
    }

    func advance(byTJ adjustment: CGFloat) {
        cursorX -= adjustment / 1000 * max(fontSize, 0.01)
    }

    // MARK: 输出

    func show(_ bytes: [UInt8]) {
        guard let decoder, !bytes.isEmpty else { return }
        let text = decoder.decode(bytes)
        guard !text.isEmpty else { return }
        append(text)
    }

    private func append(_ text: String) {
        guard outputCount < PDFTextExtractor.maxCharactersPerPage else { return }

        if !output.isEmpty {
            let dy = abs(cursorY - (lastRunY ?? cursorY))
            let lineHeight = max(1, fontSize * 0.6)
            if dy > lineHeight {
                let separator = dy > fontSize * 1.6 ? "\n\n" : "\n"
                if !output.hasSuffix(separator) {
                    if separator == "\n\n", output.hasSuffix("\n") {
                        output += "\n"
                        outputCount += 1
                    } else {
                        output += separator
                        outputCount += separator.count
                    }
                }
            } else if let lastRunEndX {
                let gap = cursorX - lastRunEndX
                if gap > max(1, fontSize * 0.2),
                   !(Self.isCJK(lastRunText.last) && Self.isCJK(text.first)) {
                    output += " "
                    outputCount += 1
                }
            }
        }

        let remaining = PDFTextExtractor.maxCharactersPerPage - outputCount
        if text.count > remaining {
            output += text.prefix(remaining)
            outputCount = PDFTextExtractor.maxCharactersPerPage
            return
        }
        output += text
        outputCount += text.count

        let width = estimatedWidth(text)
        lastRunEndX = cursorX + width
        lastRunY = cursorY
        lastRunText = text
        cursorX += width + advanceWidth(text)
    }

    private func estimatedWidth(_ text: String) -> CGFloat {
        let size = max(fontSize, 0.01)
        var width: CGFloat = 0
        for character in text {
            guard let scalar = character.unicodeScalars.first else { continue }
            switch scalar.value {
            case 0x3000...0x303F, 0x3400...0x4DBF, 0x4E00...0x9FFF,
                 0xF900...0xFAFF, 0xFF00...0xFFEF:
                width += size
            case 0x20:
                width += size * 0.25
            default:
                width += size * 0.5
            }
        }
        return width
    }

    /// Tc / Tw 的附加步进（只用于判断下一个文本 run 的间隙）。
    private func advanceWidth(_ text: String) -> CGFloat {
        var advance: CGFloat = 0
        for character in text {
            advance += characterSpacing
            if character == " " { advance += wordSpacing }
        }
        return advance
    }

    private static func isCJK(_ character: Character?) -> Bool {
        guard let character, let scalar = character.unicodeScalars.first else { return false }
        switch scalar.value {
        case 0x3000...0x303F, 0x3400...0x4DBF, 0x4E00...0x9FFF,
             0xF900...0xFAFF, 0xFF00...0xFFEF:
            return true
        default:
            return false
        }
    }

    // MARK: 图形状态

    func pushState() {
        guard stateStack.count < PDFTextExtractor.maxStateDepth else { return }
        stateStack.append(snapshot())
    }

    func popState() {
        guard let saved = stateStack.popLast() else { return }
        restore(saved)
    }

    private func snapshot() -> SavedState {
        SavedState(decoder: decoder,
                   fontSize: fontSize,
                   leading: leading,
                   wordSpacing: wordSpacing,
                   characterSpacing: characterSpacing,
                   lineX: lineX,
                   lineY: lineY,
                   cursorX: cursorX,
                   cursorY: cursorY,
                   lastRunEndX: lastRunEndX,
                   lastRunY: lastRunY,
                   lastRunText: lastRunText)
    }

    private func restore(_ saved: SavedState) {
        decoder = saved.decoder
        fontSize = saved.fontSize
        leading = saved.leading
        wordSpacing = saved.wordSpacing
        characterSpacing = saved.characterSpacing
        lineX = saved.lineX
        lineY = saved.lineY
        cursorX = saved.cursorX
        cursorY = saved.cursorY
        lastRunEndX = saved.lastRunEndX
        lastRunY = saved.lastRunY
        lastRunText = saved.lastRunText
    }

    // MARK: Form XObject

    func scanForm(name: String) {
        guard formDepth < PDFTextExtractor.maxFormDepth,
              formScans < PDFTextExtractor.maxFormScans,
              let pageResources = resources else { return }
        var xobjects: CGPDFDictionaryRef?
        guard CGPDFDictionaryGetDictionary(pageResources, "XObject", &xobjects),
              let xobjects else { return }
        var streamReference: CGPDFStreamRef?
        guard CGPDFDictionaryGetStream(xobjects, name, &streamReference),
              let stream = streamReference,
              let dictionary = CGPDFStreamGetDictionary(stream) else { return }
        var subtypePointer: UnsafePointer<CChar>?
        guard CGPDFDictionaryGetName(dictionary, "Subtype", &subtypePointer),
              let subtypePointer,
              String(cString: subtypePointer) == "Form" else { return }

        formScans += 1
        let savedState = snapshot()
        let savedResources = self.resources
        let savedFontDictionary = fontDictionary

        var formResourcesReference: CGPDFDictionaryRef?
        if CGPDFDictionaryGetDictionary(dictionary, "Resources", &formResourcesReference),
           let formResources = formResourcesReference {
            resources = formResources
            var fonts: CGPDFDictionaryRef?
            fontDictionary = CGPDFDictionaryGetDictionary(formResources, "Font", &fonts)
                ? fonts : nil
        }
        // Form 会切换字体资源；沿用外层 decoder 会把旧字体的 CMap 用到新内容上。
        decoder = nil

        let parent = CGPDFContentStreamCreateWithPage(page)
        let content = CGPDFContentStreamCreateWithStream(stream, resources ?? pageResources, parent)
        formDepth += 1
        let scanner = CGPDFScannerCreate(content, operatorTable,
                                         Unmanaged.passUnretained(self).toOpaque())
        _ = CGPDFScannerScan(scanner)
        formDepth -= 1

        self.resources = savedResources
        fontDictionary = savedFontDictionary
        restore(savedState)
    }
}

// MARK: - 操作符回调

private func scannerState(_ info: UnsafeMutableRawPointer?) -> PDFTextScanState? {
    guard let info else { return nil }
    return Unmanaged<PDFTextScanState>.fromOpaque(info).takeUnretainedValue()
}

private func pdfStringBytes(_ string: CGPDFStringRef) -> [UInt8]? {
    let length = CGPDFStringGetLength(string)
    guard length > 0, let pointer = CGPDFStringGetBytePtr(string) else { return nil }
    return [UInt8](UnsafeBufferPointer(start: pointer, count: length))
}

private func btCallback(_ scanner: CGPDFScannerRef, _ info: UnsafeMutableRawPointer?) {
    scannerState(info)?.beginText()
}

private func etCallback(_ scanner: CGPDFScannerRef, _ info: UnsafeMutableRawPointer?) {}

private func tfCallback(_ scanner: CGPDFScannerRef, _ info: UnsafeMutableRawPointer?) {
    guard let state = scannerState(info) else { return }
    var size: CGPDFReal = 0
    var namePointer: UnsafePointer<CChar>?
    guard CGPDFScannerPopNumber(scanner, &size),
          CGPDFScannerPopName(scanner, &namePointer), let namePointer else { return }
    state.fontSize = CGFloat(size)
    state.setFont(name: String(cString: namePointer))
}

private func tlCallback(_ scanner: CGPDFScannerRef, _ info: UnsafeMutableRawPointer?) {
    guard let state = scannerState(info) else { return }
    var value: CGPDFReal = 0
    if CGPDFScannerPopNumber(scanner, &value) {
        state.leading = CGFloat(value)
    }
}

private func tdCallback(_ scanner: CGPDFScannerRef, _ info: UnsafeMutableRawPointer?) {
    guard let state = scannerState(info) else { return }
    var ty: CGPDFReal = 0
    var tx: CGPDFReal = 0
    guard CGPDFScannerPopNumber(scanner, &ty),
          CGPDFScannerPopNumber(scanner, &tx) else { return }
    state.setLineOffset(tx: CGFloat(tx), ty: CGFloat(ty))
}

private func tdSetLeadingCallback(_ scanner: CGPDFScannerRef,
                                  _ info: UnsafeMutableRawPointer?) {
    guard let state = scannerState(info) else { return }
    var ty: CGPDFReal = 0
    var tx: CGPDFReal = 0
    guard CGPDFScannerPopNumber(scanner, &ty),
          CGPDFScannerPopNumber(scanner, &tx) else { return }
    state.leading = CGFloat(-ty)
    state.setLineOffset(tx: CGFloat(tx), ty: CGFloat(ty))
}

private func tmCallback(_ scanner: CGPDFScannerRef, _ info: UnsafeMutableRawPointer?) {
    guard let state = scannerState(info) else { return }
    var f: CGPDFReal = 0, e: CGPDFReal = 0, d: CGPDFReal = 0
    var c: CGPDFReal = 0, b: CGPDFReal = 0, a: CGPDFReal = 0
    guard CGPDFScannerPopNumber(scanner, &f),
          CGPDFScannerPopNumber(scanner, &e),
          CGPDFScannerPopNumber(scanner, &d),
          CGPDFScannerPopNumber(scanner, &c),
          CGPDFScannerPopNumber(scanner, &b),
          CGPDFScannerPopNumber(scanner, &a) else { return }
    state.setTextMatrix(e: CGFloat(e), f: CGFloat(f))
}

private func tstarCallback(_ scanner: CGPDFScannerRef, _ info: UnsafeMutableRawPointer?) {
    scannerState(info)?.nextLine()
}

private func tjCallback(_ scanner: CGPDFScannerRef, _ info: UnsafeMutableRawPointer?) {
    guard let state = scannerState(info) else { return }
    var string: CGPDFStringRef?
    guard CGPDFScannerPopString(scanner, &string), let string,
          let bytes = pdfStringBytes(string) else { return }
    state.show(bytes)
}

private func quoteCallback(_ scanner: CGPDFScannerRef, _ info: UnsafeMutableRawPointer?) {
    guard let state = scannerState(info) else { return }
    var string: CGPDFStringRef?
    guard CGPDFScannerPopString(scanner, &string), let string,
          let bytes = pdfStringBytes(string) else { return }
    state.nextLine()
    state.show(bytes)
}

private func doubleQuoteCallback(_ scanner: CGPDFScannerRef,
                                 _ info: UnsafeMutableRawPointer?) {
    guard let state = scannerState(info) else { return }
    var string: CGPDFStringRef?
    var characterSpacing: CGPDFReal = 0
    var wordSpacing: CGPDFReal = 0
    guard CGPDFScannerPopString(scanner, &string),
          CGPDFScannerPopNumber(scanner, &characterSpacing),
          CGPDFScannerPopNumber(scanner, &wordSpacing), let string,
          let bytes = pdfStringBytes(string) else { return }
    state.characterSpacing = CGFloat(characterSpacing)
    state.wordSpacing = CGFloat(wordSpacing)
    state.nextLine()
    state.show(bytes)
}

private func tjArrayCallback(_ scanner: CGPDFScannerRef, _ info: UnsafeMutableRawPointer?) {
    guard let state = scannerState(info) else { return }
    var array: CGPDFArrayRef?
    guard CGPDFScannerPopArray(scanner, &array), let array else { return }
    let count = CGPDFArrayGetCount(array)
    for index in 0..<count {
        var string: CGPDFStringRef?
        if CGPDFArrayGetString(array, index, &string), let string,
           let bytes = pdfStringBytes(string) {
            state.show(bytes)
        } else {
            var number: CGPDFReal = 0
            if CGPDFArrayGetNumber(array, index, &number) {
                state.advance(byTJ: CGFloat(number))
            }
        }
    }
}

private func tcCallback(_ scanner: CGPDFScannerRef, _ info: UnsafeMutableRawPointer?) {
    guard let state = scannerState(info) else { return }
    var value: CGPDFReal = 0
    if CGPDFScannerPopNumber(scanner, &value) {
        state.characterSpacing = CGFloat(value)
    }
}

private func twCallback(_ scanner: CGPDFScannerRef, _ info: UnsafeMutableRawPointer?) {
    guard let state = scannerState(info) else { return }
    var value: CGPDFReal = 0
    if CGPDFScannerPopNumber(scanner, &value) {
        state.wordSpacing = CGFloat(value)
    }
}

private func qCallback(_ scanner: CGPDFScannerRef, _ info: UnsafeMutableRawPointer?) {
    scannerState(info)?.pushState()
}

private func qPopCallback(_ scanner: CGPDFScannerRef, _ info: UnsafeMutableRawPointer?) {
    scannerState(info)?.popState()
}

private func doCallback(_ scanner: CGPDFScannerRef, _ info: UnsafeMutableRawPointer?) {
    guard let state = scannerState(info) else { return }
    var namePointer: UnsafePointer<CChar>?
    guard CGPDFScannerPopName(scanner, &namePointer), let namePointer else { return }
    state.scanForm(name: String(cString: namePointer))
}

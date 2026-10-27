import Foundation
import PDFKit
import CoreGraphics
import Vision

/// 文本层并行提取的单 worker 结果，按页序合并回主流程。
private struct VisionTextLayerOutcome: Sendable {
    var texts: [(index: Int, text: String)] = []
    var ocrPages: [Int] = []
    var textSeconds = 0.0
}

/// 第二个 PDF 方案（`rag_pdf_engine=vision`）：
/// 优先使用 PDFKit 文本层；每页有效字符数不足阈值时才渲染 + 归一化 + Apple Vision OCR。
///
/// 不依赖任何 Python/第三方 OCR；页级并发 4，结果按页序拼接。
public final class VisionPDFParser: DocumentParser {

    public let name = "VisionOCR"
    public let supportedExtensions = ["pdf"]

    /// 每页文本层可用阈值（与 Ovis 路径统一，保证文字版 PDF 一致走 PDFKit）。
    public static let textLayerMinCharacters = PDFPageRenderer.textLayerMinCharacters

    public init() {}

    public func parse(url: URL,
                      options: ParserOptions,
                      progress: ((ParserProgress) -> Void)?) async throws -> ParsedDocument {
        guard let document = PDFDocument(url: url) else {
            throw AgentError.parse("无法打开 PDF：\(url.lastPathComponent)")
        }
        if document.isLocked, !document.unlock(withPassword: "") {
            throw AgentError.parse("PDF 已加密且无法用空密码解锁：\(url.lastPathComponent)")
        }

        let totalPages = document.pageCount
        guard totalPages > 0 else {
            throw AgentError.parse("PDF 不包含任何页面：\(url.lastPathComponent)")
        }
        let pageLimit = options.maxPages > 0 ? min(totalPages, options.maxPages) : totalPages

        var pageTexts = [String?](repeating: nil, count: pageLimit)
        var pagesNeedingOCR: [Int] = []
        var textSeconds = 0.0
        var renderSeconds = 0.0
        var ocrSeconds = 0.0

        // 1. 按页并行提取文本层，并判断是否达到阈值。
        //    每个 worker 独占一个 PDFDocument（PDFKit 不能跨线程共享同一文档），
        //    文本提取是 CPU 单线程工作，串行时大 PDF 只有一个核在跑。
        let textWorkerCount = max(1, min(PDFPageRenderer.renderWorkers, pageLimit))
        let textOutcomes = try await withThrowingTaskGroup(of: VisionTextLayerOutcome.self) { group in
            for worker in 0..<textWorkerCount {
                group.addTask {
                    guard let workerDocument = PDFDocument(url: url) else {
                        throw AgentError.parse("无法打开 PDF：\(url.lastPathComponent)")
                    }
                    if workerDocument.isLocked,
                       !workerDocument.unlock(withPassword: "") {
                        throw AgentError.parse(
                            "PDF 已加密且无法用空密码解锁：\(url.lastPathComponent)")
                    }
                    var outcome = VisionTextLayerOutcome()
                    for index in stride(from: worker, to: pageLimit, by: textWorkerCount) {
                        if Task.isCancelled { throw AgentError.cancelled }
                        progress?(.page(index: index + 1, total: pageLimit, stage: "text"))
                        guard let page = workerDocument.page(at: index) else {
                            outcome.texts.append((index, ""))
                            continue
                        }
                        let textStarted = Date()
                        let text = (page.string ?? "")
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                        outcome.textSeconds += Date().timeIntervalSince(textStarted)
                        if DocumentCleaner.effectiveCharacterCount(text)
                            >= Self.textLayerMinCharacters {
                            outcome.texts.append((index, text))
                        } else {
                            outcome.ocrPages.append(index)
                        }
                    }
                    return outcome
                }
            }
            var all: [VisionTextLayerOutcome] = []
            for try await outcome in group {
                all.append(outcome)
            }
            return all
        }

        for outcome in textOutcomes {
            textSeconds += outcome.textSeconds
            for (index, text) in outcome.texts { pageTexts[index] = text }
            pagesNeedingOCR.append(contentsOf: outcome.ocrPages)
        }
        pagesNeedingOCR.sort()

        // 2. 扫描页：多 worker 并行产图（提取/渲染）+ Vision OCR 流水线重叠，
        //    结果按页序收集。空白页直接置空。
        let outcomes = try await PDFPageScanRunner.run(
            url: url,
            pages: pagesNeedingOCR,
            pageTotal: pageLimit,
            targetLongSide: OCRImagePreprocessor.maxLongSide,
            dpi: options.renderDPI,
            progress: progress) { pageIndex, image in
                progress?(.page(index: pageIndex + 1, total: pageLimit, stage: "ocr"))
                let result = try await Self.recognizeSingle(
                    image: image,
                    preprocess: options.ocrPreprocess,
                    binarize: options.ocrBinarize,
                    languages: options.ocrLanguages)
                return (text: result.text, needsVLM: false, busySeconds: result.busySeconds)
            }
        for outcome in outcomes {
            pageTexts[outcome.index] = outcome.text ?? ""
            ocrSeconds += outcome.ocrSeconds
        }
        renderSeconds = outcomes.reduce(0) { $0 + $1.renderSeconds }

        let sections = pageTexts.enumerated().flatMap { (index, text) -> [String] in
            let body = (text ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            if body.isEmpty {
                return [DocumentCleaner.pageMarker(index + 1)]
            }
            return [DocumentCleaner.pageMarker(index + 1), body]
        }

        guard pageTexts.contains(where: { !($0 ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else {
            throw AgentError.parse("未能从 PDF 提取任何文本：\(url.lastPathComponent)")
        }

        var plain = sections.joined(separator: "\n\n")
        var outlineEntries = PDFBookmarks.entries(of: document)
        var metadata: [String: String] = ["engine": "vision",
                                          "ocr_pages": String(pagesNeedingOCR.count),
                                          "timing_text": String(format: "%.2f", textSeconds),
                                          "timing_render": String(format: "%.2f", renderSeconds),
                                          "timing_ocr": String(format: "%.2f", ocrSeconds)]
        // OCR 输出是无标记纯文本：用启发式还原标题结构（目录块/编号/格式对比），
        // 转成 Markdown 后标题才能进入大纲与切片 heading path。
        if !PlainTextStructure.isMarkdownLike(plain),
           let restored = PlainTextStructure.restore(plain) {
            plain = restored.markdown
            outlineEntries.append(contentsOf: restored.entries)
            metadata["structure_restored"] = "true"
        }
        var parsed = ParsedDocument(text: plain,
                                    source: url.lastPathComponent,
                                    outline: MarkdownOutline.build(outlineEntries),
                                    metadata: metadata,
                                    pageCount: pageLimit)
        parsed = finish(parsed, options: options, profile: .ocr)
        return parsed
    }

    // MARK: - 并发 OCR

    /// 运行时覆盖闸门（rag_vision_concurrency > 0 时生效）。
    /// 必须在首次 OCR 前设置——visionGate 在首次访问时才惰性初始化。
    static var requestedConcurrency: Int?

    /// 跨文件的 Vision OCR 全局并发上限。实测 4 路已接近吞吐拐点
    /// （识别跑在 ANE/GPU，请求过多反而排队变慢），默认固定 4；
    /// rag_vision_concurrency 可覆盖用于实验。
    static var visionConcurrency: Int {
        requestedConcurrency ?? 4
    }
    private static let visionGate = AsyncSemaphore(value: visionConcurrency)

    static func recognizeBatch(_ images: [PDFPageImage],
                                preprocess: Bool,
                                binarize: Bool,
                                languages: [String]) async throws -> [(Int, String)] {
        try await withThrowingTaskGroup(of: (Int, String).self) { group in
            for item in images {
                group.addTask {
                    let result = try await Self.recognizeSingle(
                        image: item.image,
                        preprocess: preprocess,
                        binarize: binarize,
                        languages: languages)
                    return (item.pageIndex, result.text)
                }
            }

            var results: [(Int, String)] = []
            results.reserveCapacity(images.count)
            for try await result in group {
                results.append(result)
            }
            return results.sorted { $0.0 < $1.0 }
        }
    }

    /// 单页识别：预处理（归一化/锐化/可选二值化）在闸门外做（纯 CPU），
    /// Vision OCR 受全局并发闸门约束；单页失败返回空文本，取消向上传播。
    /// 返回 `busySeconds` 为闸门内实际识别耗时（不含排队等待）。
    static func recognizeSingle(image: CGImage,
                                preprocess: Bool,
                                binarize: Bool,
                                languages: [String]) async throws -> (text: String, busySeconds: Double) {
        do {
            let source = preprocess
                ? (OCRImagePreprocessor.normalize(image, binarize: binarize) ?? image)
                : image
            return try await Self.visionGate.withPermit {
                let started = Date()
                let text = try Self.recognizeText(image: source, languages: languages)
                return (text, Date().timeIntervalSince(started))
            }
        } catch {
            // 用户停止同步时必须向上传播取消，不能把剩余页当成失败页。
            if Task.isCancelled { throw AgentError.cancelled }
            // 单页失败不影响整份文档：记录并返回空页。
            AgentLog.warning("Vision OCR 失败，已跳过该页: \(error.localizedDescription)")
            return (text: "", busySeconds: 0)
        }
    }

    // MARK: - Apple Vision

    /// 供 PDF 与扫描版 EPUB 共用的 Vision OCR 入口。
    static func recognizeText(image: CGImage, languages: [String]) throws -> String {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true

        if !languages.isEmpty,
           let supported = try? request.supportedRecognitionLanguages() {
            let filtered = languages.filter { supported.contains($0) }
            if !filtered.isEmpty {
                request.recognitionLanguages = filtered
            }
        }

        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        try handler.perform([request])

        let observations = request.results ?? []
        return layoutText(from: observations)
    }

    private struct RecognizedLine {
        let text: String
        let box: CGRect
    }

    /// 阅读顺序：上 → 下、左 → 右；同一视觉行按 x 合并，段落间距保留空行。
    private static func layoutText(from observations: [VNRecognizedTextObservation]) -> String {
        let lines: [RecognizedLine] = observations.compactMap { observation in
            guard let candidate = observation.topCandidates(1).first else { return nil }
            let text = candidate.string.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return nil }
            return RecognizedLine(text: text, box: observation.boundingBox)
        }
        guard !lines.isEmpty else { return "" }

        let heights = lines.map(\.box.height).sorted()
        let medianHeight = heights[heights.count / 2]
        let lineTolerance = max(0.006, CGFloat(medianHeight) * 0.6)

        let sorted = lines.sorted { lhs, rhs in
            if abs(lhs.box.midY - rhs.box.midY) > lineTolerance {
                return lhs.box.midY > rhs.box.midY
            }
            return lhs.box.minX < rhs.box.minX
        }

        // 将 y 中心接近的观测聚合成一个文本行。
        var groups: [[RecognizedLine]] = []
        for line in sorted {
            if let last = groups.last,
               abs(last[0].box.midY - line.box.midY) <= lineTolerance {
                groups[groups.count - 1].append(line)
            } else {
                groups.append([line])
            }
        }

        let paragraphGap = max(0.012, CGFloat(medianHeight) * 0.8)
        var paragraphs: [String] = []
        var current: [String] = []
        var previousBottom: CGFloat?

        for group in groups {
            let orderedGroup = group.sorted { $0.box.minX < $1.box.minX }
            let text = joinLineTexts(orderedGroup.map(\.text))
            guard !text.isEmpty else { continue }

            if let previousBottom,
               let top = orderedGroup.map(\.box.maxY).max(),
               previousBottom - top > paragraphGap {
                paragraphs.append(current.joined(separator: "\n"))
                current = []
            }
            current.append(text)
            previousBottom = orderedGroup.map(\.box.minY).min()
        }
        if !current.isEmpty {
            paragraphs.append(current.joined(separator: "\n"))
        }

        let result = paragraphs.filter { !$0.isEmpty }.joined(separator: "\n\n")
        return removeCJKSpacing(result)
    }

    private static func joinLineTexts(_ texts: [String]) -> String {
        var result = ""
        for text in texts where !text.isEmpty {
            if result.isEmpty {
                result = text
                continue
            }
            if let last = result.last, let first = text.first, isCJK(last) && isCJK(first) {
                result += text
            } else {
                result += " " + text
            }
        }
        return result
    }

    private static func isCJK(_ character: Character) -> Bool {
        guard let scalar = character.unicodeScalars.first else { return false }
        switch scalar.value {
        case 0x3000...0x303F, 0x3400...0x4DBF, 0x4E00...0x9FFF,
             0xF900...0xFAFF, 0xFF00...0xFFEF:
            return true
        default:
            return false
        }
    }

    /// Vision 对中文常把每个字识别为独立观察；去掉汉字之间多余的空格。
    private static func removeCJKSpacing(_ text: String) -> String {
        text.replacingOccurrences(of: #"(?<=\p{Han})[ \t]+(?=\p{Han})"#,
                                  with: "",
                                  options: .regularExpression)
    }
}

// MARK: - 图像归一化 / 二值化

/// 对应 Python `preprocess_for_ocr`：
/// 长边归一化到 1500 以内 → 灰度 → 1%/99% 对比度拉伸 → 3×3 锐化 → 可选 Otsu 二值化。
enum OCRImagePreprocessor {

    static let maxLongSide = 1500

    static func normalize(_ image: CGImage, binarize: Bool) -> CGImage? {
        let sourceWidth = image.width
        let sourceHeight = image.height
        guard sourceWidth > 0, sourceHeight > 0 else { return nil }

        let longSide = max(sourceWidth, sourceHeight)
        let factor = longSide > maxLongSide
            ? Double(maxLongSide) / Double(longSide)
            : 1.0
        let width = max(1, Int((Double(sourceWidth) * factor).rounded()))
        let height = max(1, Int((Double(sourceHeight) * factor).rounded()))

        var pixels = [UInt8](repeating: 255, count: width * height)
        guard let context = CGContext(data: &pixels,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: width,
                                      space: CGColorSpaceCreateDeviceGray(),
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue) else {
            return nil
        }
        context.interpolationQuality = .high
        context.setFillColor(gray: 1, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        contrastStretch(&pixels)
        sharpen(&pixels, width: width, height: height)
        if binarize {
            otsuBinarize(&pixels)
        }
        return context.makeImage()
    }

    /// 1% / 99% 分位线性拉伸，提升低对比度扫描件的可读性。
    private static func contrastStretch(_ pixels: inout [UInt8]) {
        guard !pixels.isEmpty else { return }
        var histogram = [Int](repeating: 0, count: 256)
        for value in pixels {
            histogram[Int(value)] += 1
        }

        let total = pixels.count
        let lowTarget = max(1, Int(Double(total) * 0.01))
        let highTarget = max(1, Int(Double(total) * 0.01))

        var cumulative = 0
        var low = 0
        for value in 0..<256 {
            cumulative += histogram[value]
            if cumulative >= lowTarget {
                low = value
                break
            }
        }

        cumulative = 0
        var high = 255
        for value in stride(from: 255, through: 0, by: -1) {
            cumulative += histogram[value]
            if cumulative >= highTarget {
                high = value
                break
            }
        }

        guard high > low + 10 else { return }
        let scale = 255.0 / Double(high - low)
        for index in pixels.indices {
            let value = (Double(Int(pixels[index]) - low) * scale).rounded()
            pixels[index] = UInt8(max(0, min(255, Int(value))))
        }
    }

    /// 3×3 反锐化掩模：中心 5，四邻 -1。
    private static func sharpen(_ pixels: inout [UInt8], width: Int, height: Int) {
        guard width > 2, height > 2 else { return }
        let source = pixels
        for y in 1..<(height - 1) {
            let row = y * width
            for x in 1..<(width - 1) {
                let center = Int(source[row + x])
                let value = 5 * center
                    - Int(source[row - width + x])
                    - Int(source[row + width + x])
                    - Int(source[row + x - 1])
                    - Int(source[row + x + 1])
                pixels[row + x] = UInt8(max(0, min(255, value)))
            }
        }
    }

    /// Otsu 全局阈值二值化。
    private static func otsuBinarize(_ pixels: inout [UInt8]) {
        guard !pixels.isEmpty else { return }
        var histogram = [Int](repeating: 0, count: 256)
        for value in pixels {
            histogram[Int(value)] += 1
        }

        let total = pixels.count
        var sum = 0.0
        for value in 0..<256 {
            sum += Double(value * histogram[value])
        }

        var sumBackground = 0.0
        var weightBackground = 0
        var bestVariance = 0.0
        var threshold = 127

        for value in 0..<256 {
            weightBackground += histogram[value]
            if weightBackground == 0 { continue }
            let weightForeground = total - weightBackground
            if weightForeground == 0 { break }

            sumBackground += Double(value * histogram[value])
            let meanBackground = sumBackground / Double(weightBackground)
            let meanForeground = (sum - sumBackground) / Double(weightForeground)
            let variance = Double(weightBackground) * Double(weightForeground)
                * (meanBackground - meanForeground) * (meanBackground - meanForeground)
            if variance > bestVariance {
                bestVariance = variance
                threshold = value
            }
        }

        let cutoff = UInt8(threshold)
        for index in pixels.indices {
            pixels[index] = pixels[index] > cutoff ? 255 : 0
        }
    }
}

// MARK: - PDF 书签大纲

/// 从 PDF 自带书签（outline）提取章节大纲。制作规范的 PDF 书签就是完整目录，
/// 比从正文正则猜测可靠得多；没有书签时返回空，回退到正文识别。
enum PDFBookmarks {
    static func entries(of document: PDFDocument) -> [OutlineEntry] {
        guard let root = document.outlineRoot, root.numberOfChildren > 0 else { return [] }
        var entries: [OutlineEntry] = []

        func walk(_ item: PDFOutline, level: Int) {
            if let label = item.label?.trimmingCharacters(in: .whitespacesAndNewlines),
               !label.isEmpty, label.count <= 100 {
                entries.append(OutlineEntry(level: level, title: label))
            }
            for index in 0..<item.numberOfChildren {
                if let child = item.child(at: index) { walk(child, level: level + 1) }
            }
        }
        // root 自身的 label 是文档名，直接从子节点开始。
        for index in 0..<root.numberOfChildren {
            if let child = root.child(at: index) { walk(child, level: 1) }
        }
        return entries
    }
}

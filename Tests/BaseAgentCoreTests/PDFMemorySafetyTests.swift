import XCTest
import CoreGraphics
import ImageIO
import PDFKit
import UniformTypeIdentifiers
@testable import BaseAgentCore

/// 大 PDF 转换的内存安全回归测试：
/// - 文件体积越大，页级 worker / 在途页数 / 批大小必须越小；
/// - 内嵌 JPEG 必须先降采样再解码，不能先解出整张超大图；
/// - 目录转换时大文件必须单独成批，避免多份大 PDF 同时驻留。
final class PDFMemorySafetyTests: XCTestCase {

    private let mb: Int64 = 1024 * 1024

    // MARK: - 页级处理预算

    func testVeryLargePDFLimitsWorkersInFlightAndBatch() {
        let plan = PDFProcessingPlan.plan(fileSizeBytes: 300 * mb, renderWorkers: 4)

        XCTAssertLessThanOrEqual(plan.workerCount, 2,
                                 "300MB 级 PDF 不应再开 4 个页级 worker")
        XCTAssertLessThanOrEqual(plan.inFlightPerWorker, 2,
                                 "大文件每个 worker 的在途页数必须收紧")
        XCTAssertLessThanOrEqual(plan.pageBatchSize, 8,
                                 "大文件必须用更小的批次，避免整批图像驻留")
        XCTAssertGreaterThanOrEqual(plan.workerCount, 1)
        XCTAssertGreaterThanOrEqual(plan.inFlightPerWorker, 1)
        XCTAssertGreaterThanOrEqual(plan.pageBatchSize, 1)
    }

    func testMediumPDFUsesModerateWindow() {
        let plan = PDFProcessingPlan.plan(fileSizeBytes: 80 * mb, renderWorkers: 4)

        XCTAssertLessThanOrEqual(plan.workerCount, 3)
        XCTAssertLessThanOrEqual(plan.inFlightPerWorker, 3)
        XCTAssertLessThanOrEqual(plan.pageBatchSize, 24)
        XCTAssertGreaterThanOrEqual(plan.workerCount, 1)
    }

    func testSmallPDFKeepsDefaultConcurrency() {
        let plan = PDFProcessingPlan.plan(fileSizeBytes: 2 * mb, renderWorkers: 4)

        XCTAssertEqual(plan.workerCount, 4)
        XCTAssertEqual(plan.inFlightPerWorker, 4)
        XCTAssertEqual(plan.pageBatchSize, 50)
    }

    func testPlanNeverExceedsRequestedWorkers() {
        for size in [0, 1, 16, 64, 192, 512].map({ Int64($0) * mb }) {
            for workers in 1...8 {
                let plan = PDFProcessingPlan.plan(fileSizeBytes: size,
                                                  renderWorkers: workers)
                XCTAssertLessThanOrEqual(plan.workerCount, workers, "size=\(size)")
            }
        }
    }

    // MARK: - 内嵌 JPEG 解码

    func testEmbeddedJPEGIsDownsampledBeforeDecode() throws {
        let data = try makeJPEG(width: 4_000, height: 3_000)
        let image = try XCTUnwrap(
            PDFPageRenderer.decodeEmbeddedJPEG(data, targetLongSide: 1_000),
            "有效 JPEG 应能解码")

        XCTAssertLessThanOrEqual(max(image.width, image.height), 1_000,
                                 "必须先按目标长边降采样，不能先解出 4000×3000 原图")
        XCTAssertLessThan(image.width * image.height, 4_000 * 3_000,
                          "解码结果必须小于原始像素数")
    }

    func testEmbeddedJPEGKeepsAspectRatioWhenDownsampled() throws {
        let data = try makeJPEG(width: 4_800, height: 1_200)
        let image = try XCTUnwrap(
            PDFPageRenderer.decodeEmbeddedJPEG(data, targetLongSide: 1_000))

        let ratio = Double(image.width) / Double(image.height)
        XCTAssertEqual(ratio, 4.0, accuracy: 0.05,
                       "降采样不能改变页面宽高比（否则 OCR 版面会变形）")
    }

    func testPDFWithHugeEmbeddedJPEGIsExtractedAtTargetSize() throws {
        let jpeg = try makeJPEG(width: 4_000, height: 3_000)
        let pdf = makeSingleImagePDF(jpeg: jpeg,
                                     pixelWidth: 4_000, pixelHeight: 3_000)
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pdf-image-\(UUID().uuidString).pdf")
        try pdf.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let document = try XCTUnwrap(CGPDFDocument(url as CFURL),
                                     "合成 PDF 应能被 CGPDFDocument 打开")
        let page = try XCTUnwrap(document.page(at: 1))
        let image = try XCTUnwrap(
            PDFPageRenderer.extractEmbeddedJPEG(cgPage: page, targetLongSide: 1_000),
            "整页 DCTDecode JPEG 应该走直提取路径")

        XCTAssertLessThanOrEqual(max(image.width, image.height), 1_000)
        XCTAssertGreaterThan(max(image.width, image.height), 900,
                             "降采样应保留目标分辨率，不能缩成缩略图")
    }

    func testEmbeddedJPEGRejectedWhenAbovePixelSafetyLimit() throws {
        let data = try makeJPEG(width: 1_000, height: 800)
        // 用极小上限模拟超高像素/解压炸弹图片：必须放弃直提取，
        // 让调用方回退到有界的页面栅格化路径。
        let image = PDFPageRenderer.decodeEmbeddedJPEG(
            data, targetLongSide: 800,
            maxPixelCount: 100_000)
        XCTAssertNil(image)
    }

    // MARK: - 图片密集页的文本层提取

    /// 图片 XObject 很多的页面必须跳过 PDFKit 文本行布局，
    /// 否则 `PDFPage.string` 会在 CRCTLD 空白区域分析里爆炸。
    func testImageHeavyPageUsesBoundedTextScanner() throws {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pdf-image-heavy-\(UUID().uuidString).pdf")
        defer { try? FileManager.default.removeItem(at: url) }
        try makeTextAndImagePDF(at: url,
                                text: Self.textLayerSample,
                                imageCount: PDFTextExtractor.imageHeavyThreshold + 8)

        let document = try XCTUnwrap(PDFDocument(url: url))
        let page = try XCTUnwrap(document.page(at: 0))
        let cgPage = try XCTUnwrap(page.pageRef)

        let imageCount = PDFTextExtractor.imageObjectCount(for: cgPage)
        XCTAssertGreaterThanOrEqual(imageCount, PDFTextExtractor.imageHeavyThreshold,
                                    "合成页面应包含足够多的图片 XObject")

        let extraction = PDFTextExtractor.text(for: page)
        XCTAssertTrue(extraction.usedScanner, "图片密集页不能调用 PDFPage.string")
        XCTAssertTrue(extraction.text.contains("image-heavy layout"), extraction.text)
    }

    /// 常见整页扫描图（每页 1 张 4000×3000）必须继续走 PDFKit：
    /// 实测这类页面 PDFKit 提取 30 页仅 0.06s，改走 CGPDFScanner 会慢
    /// 30～50 倍。只有极端大图才需要扫描器兜底。
    func testNormalFullPageImageKeepsPDFKitExtraction() throws {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pdf-single-large-image-\(UUID().uuidString).pdf")
        defer { try? FileManager.default.removeItem(at: url) }
        try makeTextAndImagePDF(at: url,
                                text: Self.textLayerSample,
                                imageCount: 1,
                                imageWidth: 4_000,
                                imageHeight: 3_000)

        let document = try XCTUnwrap(PDFDocument(url: url))
        let page = try XCTUnwrap(document.page(at: 0))
        let cgPage = try XCTUnwrap(page.pageRef)

        let profile = PDFTextExtractor.imageProfile(for: cgPage)
        XCTAssertEqual(profile.count, 1)
        XCTAssertLessThan(profile.largestPixelCount,
                          PDFTextExtractor.riskySingleImagePixels,
                          "1200 万像素仍属于常见扫描图，不应触发扫描器")

        let extraction = PDFTextExtractor.text(for: page)
        XCTAssertFalse(extraction.usedScanner,
                       "常见整页图必须走更快的 PDFKit 文本层")
        XCTAssertTrue(extraction.text.contains("image-heavy layout"), extraction.text)
    }

    /// 单张极端大图仍要兜底走扫描器。
    func testExtremeSingleImageProfileUsesBoundedScanner() {
        let profile = PDFTextExtractor.ImageProfile(
            count: 1,
            pixelCount: PDFTextExtractor.riskySingleImagePixels,
            largestPixelCount: PDFTextExtractor.riskySingleImagePixels)
        XCTAssertTrue(profile.isMemoryRisky)
    }

    /// 中等图片总量不大时继续走 PDFKit；只有总像素到极端值才改走扫描器。
    func testMediumImageGalleryKeepsPDFKitExtraction() throws {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pdf-many-medium-images-\(UUID().uuidString).pdf")
        defer { try? FileManager.default.removeItem(at: url) }
        try makeTextAndImagePDF(at: url,
                                text: Self.textLayerSample,
                                imageCount: 10,
                                imageWidth: 1_200,
                                imageHeight: 900)

        let document = try XCTUnwrap(PDFDocument(url: url))
        let page = try XCTUnwrap(document.page(at: 0))
        let cgPage = try XCTUnwrap(page.pageRef)

        let profile = PDFTextExtractor.imageProfile(for: cgPage)
        XCTAssertEqual(profile.count, 10)
        XCTAssertLessThan(profile.pixelCount,
                          PDFTextExtractor.riskyTotalImagePixels,
                          "10 张 1.08MP 图片的提取成本远低于扫描器，不应改道")

        let extraction = PDFTextExtractor.text(for: page)
        XCTAssertFalse(extraction.usedScanner,
                       "总像素未到极端值的页面应走 PDFKit")
    }

    func testExtremeTotalImageProfileUsesBoundedScanner() {
        let profile = PDFTextExtractor.ImageProfile(
            count: 10,
            pixelCount: PDFTextExtractor.riskyTotalImagePixels,
            largestPixelCount: 8_000_000)
        XCTAssertTrue(profile.isMemoryRisky)
    }

    /// 小图标/小装饰图不影响阅读顺序，继续走 PDFKit，避免无谓降低文本质量。
    func testSmallImagePageKeepsPDFKitExtraction() throws {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pdf-small-image-\(UUID().uuidString).pdf")
        defer { try? FileManager.default.removeItem(at: url) }
        try makeTextAndImagePDF(at: url,
                                text: Self.textLayerSample,
                                imageCount: 1,
                                imageWidth: 64,
                                imageHeight: 64)

        let document = try XCTUnwrap(PDFDocument(url: url))
        let page = try XCTUnwrap(document.page(at: 0))
        let cgPage = try XCTUnwrap(page.pageRef)

        let profile = PDFTextExtractor.imageProfile(for: cgPage)
        XCTAssertEqual(profile.count, 1)
        XCTAssertLessThan(profile.pixelCount,
                          PDFTextExtractor.riskySingleImagePixels)

        let extraction = PDFTextExtractor.text(for: page)
        XCTAssertFalse(extraction.usedScanner,
                       "小图页应继续使用 PDFKit 文本层")
        XCTAssertTrue(extraction.text.contains("image-heavy layout"), extraction.text)
    }

    /// 低图片页继续使用 PDFKit，保持原有阅读顺序与分段行为。
    func testLowImagePageKeepsPDFKitExtraction() throws {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pdf-text-only-\(UUID().uuidString).pdf")
        defer { try? FileManager.default.removeItem(at: url) }
        try makeTextAndImagePDF(at: url, text: Self.textLayerSample, imageCount: 0)

        let document = try XCTUnwrap(PDFDocument(url: url))
        let page = try XCTUnwrap(document.page(at: 0))
        let cgPage = try XCTUnwrap(page.pageRef)

        XCTAssertEqual(PDFTextExtractor.imageObjectCount(for: cgPage), 0)
        let extraction = PDFTextExtractor.text(for: page)
        XCTAssertFalse(extraction.usedScanner, "普通文字页应继续走 PDFKit 文本层")
        XCTAssertTrue(extraction.text.contains("image-heavy layout"), extraction.text)
    }

    /// 解析器集成：图片密集的文字页不应要求 VLM，也不应回退 OCR。
    func testOvisParserExtractsImageHeavyTextPageWithoutVLM() throws {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pdf-image-heavy-parse-\(UUID().uuidString).pdf")
        defer { try? FileManager.default.removeItem(at: url) }
        try makeTextAndImagePDF(at: url,
                                text: Self.textLayerSample,
                                imageCount: PDFTextExtractor.imageHeavyThreshold + 8)

        let parsed = try awaitSync {
            try await OvisPDFParser(vlm: nil).parse(
                url: url,
                options: ParserOptions(pdfEngine: .ovis),
                progress: nil)
        }
        XCTAssertEqual(parsed.metadata["scanner_pages"], "1")
        XCTAssertEqual(parsed.metadata["text_pages"], "1")
        XCTAssertEqual(parsed.metadata["ocr_pages"], "0")
        XCTAssertTrue(parsed.text.contains("image-heavy layout"), parsed.text)
    }

    /// 真实大文件回归（可选）：`BASE_AGENT_PDF_MEMORY_TEST=/path/to.pdf`
    /// 指向 300MB 级图片密集 PDF，完整走一遍 Ovis 文本层提取。
    /// 修复前会在第 18 页附近涨到数十 GB 内存并卡死。
    func testRealPDFTextLayerStaysBoundedWhenEnabled() throws {
        guard let path = ProcessInfo.processInfo.environment["BASE_AGENT_PDF_MEMORY_TEST"],
              !path.isEmpty else {
            throw XCTSkip("设置 BASE_AGENT_PDF_MEMORY_TEST 指向测试 PDF 后运行")
        }
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("测试 PDF 不存在：\(path)")
        }

        let parsed = try awaitSync(timeout: 900) {
            try await OvisPDFParser(vlm: nil).parse(
                url: url,
                options: ParserOptions(pdfEngine: .ovis),
                progress: nil)
        }
        XCTAssertGreaterThan(parsed.text.count, 10_000,
                             "真实 PDF 应提取出完整正文")
        XCTAssertGreaterThan(Int(parsed.metadata["scanner_pages"] ?? "0") ?? 0, 0,
                             "图片密集页应走有界扫描器")
    }

    /// 真实大文件走「转换工具」完整链路（默认 vision 引擎）：
    /// 写 Markdown、苹果 Vision 兜底扫描页，验证不因文本层布局爆内存。
    func testRealPDFConversionToolStaysBoundedWhenEnabled() throws {
        guard let path = ProcessInfo.processInfo.environment["BASE_AGENT_PDF_MEMORY_TEST"],
              !path.isEmpty else {
            throw XCTSkip("设置 BASE_AGENT_PDF_MEMORY_TEST 指向测试 PDF 后运行")
        }
        let source = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("测试 PDF 不存在：\(path)")
        }
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pdf-real-convert-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let output = root.appendingPathComponent("out")

        var config = AgentConfig()
        config.ragPDFEngine = .vision
        let report = try awaitSync(timeout: 1_800) {
            try await MarkdownConversionService().convert(
                request: MarkdownConversionRequest(sourceURL: source,
                                                   outputDirectory: output,
                                                   config: config))
        }

        XCTAssertFalse(report.wasCancelled)
        XCTAssertEqual(report.succeededCount, 1,
                       report.results.first?.error ?? "真实 PDF 转换应成功")
        let markdownURL = try XCTUnwrap(report.results.first?.outputURL)
        let markdown = try String(contentsOf: markdownURL, encoding: .utf8)
        XCTAssertGreaterThan(markdown.count, 10_000)
        XCTAssertTrue(markdown.contains("课堂讲解"), "图片密集页的正文应完整保留")
        // 第 18 页是 TimesNewRoman + WinAnsi 的简单字体正文，扫描器必须
        // 按单字节字符码解码，否则 "Excel 2016" 会被两两拼接而整段丢失。
        XCTAssertTrue(markdown.contains("对于Excel 2016的初学者"),
                      "图片密集页的英文正文不应丢字")
    }

    // MARK: - 目录转换分批

    func testConversionBatchesIsolateHeavyFiles() {
        let batches = MarkdownConversionService.batchIndices(
            fileSizes: [300 * mb, 300 * mb, 1 * mb, 1 * mb, 1 * mb],
            baseConcurrency: 6)

        XCTAssertEqual(batches, [[0], [1], [2, 3, 4]],
                       "大文件必须各自单独成批，轻文件才允许并发")
    }

    func testConversionBatchesCapLightFileConcurrency() {
        let batches = MarkdownConversionService.batchIndices(
            fileSizes: Array(repeating: 1 * mb, count: 5),
            baseConcurrency: 2)

        XCTAssertEqual(batches, [[0, 1], [2, 3], [4]])
    }

    func testConversionBatchesPreserveOrderWhenHeavyFileComesFirst() {
        let batches = MarkdownConversionService.batchIndices(
            fileSizes: [300 * mb, 1 * mb, 1 * mb],
            baseConcurrency: 6)

        XCTAssertEqual(batches, [[0], [1, 2]])
    }

    // MARK: - 目录并发转换回归

    func testDirectoryConversionWithConcurrencyKeepsOrderAndOutputs() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-convert-conc-\(UUID().uuidString)")
        let input = root.appendingPathComponent("input")
        let output = root.appendingPathComponent("out")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        for index in 0..<12 {
            try "并发转换测试内容 \(index)".write(
                to: input.appendingPathComponent("file-\(String(format: "%02d", index)).txt"),
                atomically: true, encoding: .utf8)
        }

        var config = AgentConfig()
        config.ragIngestConcurrency = 6
        let report = try await MarkdownConversionService().convert(
            request: MarkdownConversionRequest(sourceURL: input,
                                               outputDirectory: output,
                                               config: config))

        XCTAssertFalse(report.wasCancelled)
        XCTAssertEqual(report.results.count, 12)
        XCTAssertEqual(report.succeededCount, 12)
        XCTAssertEqual(report.results.map(\.source.lastPathComponent),
                       (0..<12).map { "file-\(String(format: "%02d", $0)).txt" },
                       "并发处理也必须按发现顺序返回结果")
        for index in 0..<12 {
            let markdown = output.appendingPathComponent(
                "file-\(String(format: "%02d", index)).md")
            XCTAssertTrue(FileManager.default.fileExists(atPath: markdown.path))
            XCTAssertTrue(try String(contentsOf: markdown, encoding: .utf8)
                .contains("并发转换测试内容 \(index)"))
        }
    }

    // MARK: - 工具

    private static let textLayerSample =
        "This page shows an image-heavy layout and still keeps a readable text layer for extraction."

    private func awaitSync<T>(timeout: TimeInterval = 60,
                              _ body: @escaping () async throws -> T) throws -> T {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<T, Error>!
        Task {
            do { result = .success(try await body()) }
            catch { result = .failure(error) }
            semaphore.signal()
        }
        guard semaphore.wait(timeout: .now() + timeout) == .success else {
            throw AgentError.parse("测试等待超时（\(timeout)s）")
        }
        return try XCTUnwrap(result).get()
    }

    /// 构造单页 PDF：一段 Helvetica 文本 + `imageCount` 个独立 DCTDecode 图片 XObject。
    /// 图片对象刻意分开编号，确保页面的 `/XObject` 字典里有真实的图片数量。
    private func makeTextAndImagePDF(at url: URL,
                                     text: String,
                                     imageCount: Int,
                                     imageWidth: Int = 64,
                                     imageHeight: Int = 64) throws {
        let jpeg = try makeJPEG(width: 64, height: 64)
        var data = Data()
        var offsets: [Int] = []

        func appendASCII(_ value: String) {
            data.append(Data(value.utf8))
        }

        func addObject(_ body: String, stream: Data? = nil) {
            offsets.append(data.count)
            appendASCII("\(offsets.count) 0 obj\n")
            appendASCII(body)
            if let stream {
                appendASCII("\nstream\n")
                data.append(stream)
                appendASCII("\nendstream")
            }
            appendASCII("\nendobj\n")
        }

        appendASCII("%PDF-1.4\n")
        appendASCII("%\u{00E2}\u{00E3}\u{00CF}\u{00D3}\n")

        addObject("<< /Type /Catalog /Pages 2 0 R >>")
        addObject("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")

        let imageObjectStart = 4
        let fontObject = imageObjectStart + imageCount
        let contentObject = fontObject + 1
        var xobjects = ""
        for index in 0..<imageCount {
            xobjects += " /Im\(index) \(imageObjectStart + index) 0 R"
        }
        addObject("""
        << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
           /Resources << /XObject <<\(xobjects) >> /Font << /F1 \(fontObject) 0 R >> >>
           /Contents \(contentObject) 0 R >>
        >>
        """)

        for _ in 0..<imageCount {
            addObject("""
            << /Type /XObject /Subtype /Image /Width \(imageWidth) /Height \(imageHeight)
               /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode
               /Length \(jpeg.count) >>
            """, stream: jpeg)
        }

        addObject("""
        << /Type /Font /Subtype /Type1 /BaseFont /Helvetica
           /Encoding /WinAnsiEncoding >>
        """)

        let escaped = text
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "(", with: "\\(")
            .replacingOccurrences(of: ")", with: "\\)")
        var content = "BT /F1 24 Tf 40 720 Td (\(escaped)) Tj ET\n"
        for index in 0..<imageCount {
            let column = index % 10
            let row = index / 10
            content += "q 32 0 0 32 \(20 + column * 48) \(520 - row * 48) cm /Im\(index) Do Q\n"
        }
        addObject("<< /Length \(content.utf8.count) >>", stream: Data(content.utf8))

        let xrefOffset = data.count
        appendASCII("xref\n0 \(offsets.count + 1)\n")
        appendASCII("0000000000 65535 f \n")
        for offset in offsets {
            appendASCII(String(format: "%010d 00000 n \n", offset))
        }
        appendASCII("""
        trailer
        << /Size \(offsets.count + 1) /Root 1 0 R >>
        startxref
        \(xrefOffset)
        %%EOF

        """)
        try data.write(to: url)
    }

    private func makeJPEG(width: Int, height: Int) throws -> Data {
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
            ?? CGColorSpaceCreateDeviceRGB()
        let context = try XCTUnwrap(CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        context.setFillColor(CGColor(red: 0.9, green: 0.9, blue: 0.9, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.setFillColor(CGColor(red: 0.1, green: 0.1, blue: 0.1, alpha: 1))
        for x in stride(from: 0, to: width, by: 80) {
            context.fill(CGRect(x: x, y: 0, width: 8, height: height))
        }
        let image = try XCTUnwrap(context.makeImage())

        let data = NSMutableData()
        let destination = try XCTUnwrap(CGImageDestinationCreateWithData(
            data, UTType.jpeg.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(
            destination, image,
            [kCGImageDestinationLossyCompressionQuality: 0.8] as CFDictionary)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return data as Data
    }

    /// 构造只含一张整页 DCTDecode JPEG 的最小 PDF（满足 CPDFShim 的
    /// 「单图 / 无蒙版 / 8bpc / 页面比例一致」条件）。
    private func makeSingleImagePDF(jpeg: Data,
                                    pixelWidth: Int,
                                    pixelHeight: Int) -> Data {
        var data = Data()
        var objectOffsets: [Int] = []

        func appendASCII(_ text: String) {
            data.append(Data(text.utf8))
        }

        appendASCII("%PDF-1.4\n")
        appendASCII("%\u{00E2}\u{00E3}\u{00CF}\u{00D3}\n")

        objectOffsets.append(data.count)
        appendASCII("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

        objectOffsets.append(data.count)
        appendASCII("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")

        let pageWidth = 800
        let pageHeight = 600
        objectOffsets.append(data.count)
        appendASCII("""
        3 0 obj
        << /Type /Page /Parent 2 0 R /MediaBox [0 0 \(pageWidth) \(pageHeight)]
           /Resources << /XObject << /Im0 4 0 R >> >>
           /Contents 5 0 R >>
        endobj

        """)

        objectOffsets.append(data.count)
        appendASCII("""
        4 0 obj
        << /Type /XObject /Subtype /Image /Width \(pixelWidth)
           /Height \(pixelHeight) /ColorSpace /DeviceRGB /BitsPerComponent 8
           /Filter /DCTDecode /Length \(jpeg.count) >>
        stream

        """)
        data.append(jpeg)
        appendASCII("\nendstream\nendobj\n")

        let content = "q \(pageWidth) 0 0 \(pageHeight) 0 0 cm /Im0 Do Q\n"
        objectOffsets.append(data.count)
        appendASCII("""
        5 0 obj
        << /Length \(content.utf8.count) >>
        stream

        """)
        appendASCII(content)
        appendASCII("endstream\nendobj\n")

        let xrefOffset = data.count
        appendASCII("xref\n0 6\n")
        appendASCII("0000000000 65535 f \n")
        for offset in objectOffsets {
            appendASCII(String(format: "%010d 00000 n \n", offset))
        }
        appendASCII("""
        trailer
        << /Size 6 /Root 1 0 R >>
        startxref
        \(xrefOffset)
        %%EOF

        """)
        return data
    }
}

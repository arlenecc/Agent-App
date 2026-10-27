import XCTest
import CoreGraphics
import ImageIO
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

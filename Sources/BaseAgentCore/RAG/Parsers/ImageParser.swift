import Foundation
import CoreGraphics
import ImageIO

/// 独立图片文件（png/jpg/heic 等）的 OCR 解析。
///
/// 默认走 Apple Vision（与扫描 PDF 的 Vision 引擎同一识别通道），
/// 识别出的纯文本经启发式结构还原转成 markdown 后入库。
public final class ImageParser: DocumentParser {
    public let name = "VisionImageOCR"
    public let supportedExtensions = [
        "png", "jpg", "jpeg", "tiff", "tif", "bmp", "heic", "heif", "webp",
    ]

    public init() {}

    public func parse(url: URL, options: ParserOptions,
                      progress: ((ParserProgress) -> Void)?) async throws -> ParsedDocument {
        progress?(.message("图片 OCR（Apple Vision）: \(url.lastPathComponent)"))
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw AgentError.parse("无法读取图片: \(url.lastPathComponent)")
        }

        let normalized = OCRImagePreprocessor.normalize(
            cgImage, binarize: options.ocrBinarize) ?? cgImage
        // Vision 单次识别；语言与 PDF 页识别保持一致（中文优先、英文兜底）。
        var raw = try VisionPDFParser.recognizeText(
            image: normalized, languages: ["zh-Hans", "en-US"])
        // 二值化可能把「深底浅字」的图片洗成整片黑，返回空；用原图兜底识别一次。
        if DocumentCleaner.effectiveCharacterCount(raw) == 0 {
            raw = try VisionPDFParser.recognizeText(
                image: cgImage, languages: ["zh-Hans", "en-US"])
        }

        guard !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AgentError.parse("图片中未识别出文本: \(url.lastPathComponent)")
        }

        // OCR 输出是无结构纯文本：还原目录/标题结构并转 markdown。
        var text = raw
        if !PlainTextStructure.isMarkdownLike(text),
           let restored = PlainTextStructure.restore(text) {
            text = restored.markdown
        }

        var document = ParsedDocument(text: text, source: url.lastPathComponent)
        document.metadata["ocr_engine"] = "apple-vision"
        document.metadata["structure_restored"] = "1"
        document.metadata["width"] = String(cgImage.width)
        document.metadata["height"] = String(cgImage.height)
        return finish(document, options: options, profile: .ocr)
    }
}

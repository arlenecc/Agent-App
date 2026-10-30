import XCTest
import AppKit
import CoreGraphics
@testable import BaseAgentCore

/// OCR 预处理回归测试。
///
/// 历史 bug：`OCRImagePreprocessor.normalize` 用 Swift 数组作为
/// `CGContext` 的位图缓冲区，却通过 `context.makeImage()` 返回结果。
/// 像素处理里的局部数组拷贝触发写时复制后，context 仍指向旧缓冲区：
/// - 串行时返回的是「画完但未预处理」的灰度图（二值化/锐化全部丢失）；
/// - 并发时旧缓冲区已被释放并复用，整页变黑，Vision 直接返回空结果，
///   解析结果表现为「随机漏页」。
final class OCRPreprocessingTests: XCTestCase {

    func testNormalizeBinarizeIsAppliedToOutput() throws {
        let image = try makeTextImage()
        let normalized = try XCTUnwrap(
            OCRImagePreprocessor.normalize(image, binarize: true))

        let pixels = try Self.pixels(of: normalized)
        XCTAssertTrue(pixels.allSatisfy { $0 == 0 || $0 == 255 },
                      "返回图必须体现 Otsu 二值化，不能是未处理的灰度图")
        XCTAssertTrue(pixels.contains(0), "文字像素应保留")
        XCTAssertTrue(pixels.contains(255), "背景像素应保留")
    }

    func testNormalizeWithoutBinarizeKeepsGrayscale() throws {
        let image = try makeTextImage()
        let normalized = try XCTUnwrap(
            OCRImagePreprocessor.normalize(image, binarize: false))

        let values = Set(try Self.pixels(of: normalized))
        XCTAssertGreaterThan(values.count, 2,
                             "非二值化模式应保留灰度层次")
    }

    /// 并发归一化必须得到完全一致且非空白的图像：
    /// 这不是单纯的线程安全问题，而是返回图不能引用已释放的临时缓冲区。
    func testConcurrentNormalizeReturnsStableImages() async throws {
        let image = try makeTextImage()
        let expected = try XCTUnwrap(
            OCRImagePreprocessor.normalize(image, binarize: true))
        let expectedPixels = try Self.pixels(of: expected)
        XCTAssertTrue(expectedPixels.contains(0))

        let results = try await withThrowingTaskGroup(of: [UInt8].self) { group in
            for _ in 0..<8 {
                group.addTask {
                    let normalized = try XCTUnwrap(
                        OCRImagePreprocessor.normalize(image, binarize: true))
                    return try Self.pixels(of: normalized)
                }
            }
            var all: [[UInt8]] = []
            for try await result in group { all.append(result) }
            return all
        }

        for (index, result) in results.enumerated() {
            XCTAssertEqual(result, expectedPixels,
                           "第 \(index) 个并发结果与串行结果不一致")
            XCTAssertTrue(result.allSatisfy { $0 == 0 || $0 == 255 },
                          "第 \(index) 个并发结果丢失了二值化")
        }
    }

    // MARK: - Helpers

    private static func pixels(of image: CGImage) throws -> [UInt8] {
        XCTAssertEqual(image.bitsPerComponent, 8)
        XCTAssertEqual(image.bitsPerPixel, 8,
                       "归一化输出应是 8bpc 单通道灰度图")
        let data = try XCTUnwrap(image.dataProvider?.data as Data?,
                                 "归一化图像必须持有自己的像素数据")
        let count = image.width * image.height
        XCTAssertEqual(data.count, count)
        return Array(data.prefix(count))
    }

    /// 生成带文字的浅色底图，尺寸足够触发多行排版与 Otsu 阈值。
    private func makeTextImage() throws -> CGImage {
        let size = NSSize(width: 900, height: 1200)
        let image = NSImage(size: size)
        image.lockFocus()
        NSColor.white.setFill()
        NSRect(origin: .zero, size: size).fill()
        let paragraph = NSMutableParagraphStyle()
        paragraph.lineSpacing = 18
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 42),
            .foregroundColor: NSColor.black,
            .paragraphStyle: paragraph,
        ]
        let lines = [
            "OCR preprocessing regression sample.",
            "The quick brown fox jumps over the lazy dog.",
            "二值化之后文字应该保留，背景应该为白色。",
            "第一行用于检测阈值，第二行用于检测排版。",
            "If the output is all black, Vision returns nothing.",
            "这是并发归一化稳定性测试的样例页面。",
            "Line 7 keeps the page from being nearly blank.",
            "Line 8 makes sure Otsu has enough text pixels.",
        ]
        for (index, line) in lines.enumerated() {
            let point = NSPoint(x: 60, y: size.height - 120 - CGFloat(index) * 90)
            line.draw(at: point, withAttributes: attributes)
        }
        image.unlockFocus()

        var rect = CGRect(origin: .zero, size: size)
        let cgImage = try XCTUnwrap(image.cgImage(forProposedRect: &rect,
                                                 context: nil, hints: nil))
        return cgImage
    }
}

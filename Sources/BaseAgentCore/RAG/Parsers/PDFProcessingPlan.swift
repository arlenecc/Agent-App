import Foundation

/// 单个 PDF 的页级处理预算。
///
/// 超大 PDF（扫描件常见 100MB～1GB）单页内嵌图片的像素数可能高得离谱，
/// 如果仍按普通文件开 4 个 worker × 4 页在途，任何一页的临时位图都会被
/// 放大 16 倍。这里按文件体积收紧 worker / 在途页 / 批大小：
/// - 体积越大，同时驻留的 `PDFDocument`、页面位图和 OCR 任务越少；
/// - 小文件保持原有并发，避免为极少数大文件牺牲整体吞吐；
/// - 页面处理仍是分批 + 流水线并发，只是窗口随文件重量自适应。
struct PDFProcessingPlan: Equatable, Sendable {
    /// ≥192MB：2 个 worker、每 worker 2 页在途、8 页一批。
    static let heavyFileBytes: Int64 = 192 * 1024 * 1024
    /// ≥64MB：3 个 worker、每 worker 3 页在途、24 页一批。
    static let mediumFileBytes: Int64 = 64 * 1024 * 1024
    /// ≥16MB：保持 worker 数，但收紧在途页与批次。
    static let largeFileBytes: Int64 = 16 * 1024 * 1024

    let workerCount: Int
    let inFlightPerWorker: Int
    let pageBatchSize: Int
    let fileSizeBytes: Int64

    var isConstrained: Bool {
        workerCount < PDFPageRenderer.renderWorkers
            || inFlightPerWorker < 4
            || pageBatchSize < 50
    }

    var summary: String {
        "\(workerCount) workers × \(inFlightPerWorker) 页在途 / \(pageBatchSize) 页一批"
    }

    static func plan(fileSizeBytes: Int64,
                     renderWorkers: Int = PDFPageRenderer.renderWorkers) -> PDFProcessingPlan {
        let workers = max(1, renderWorkers)
        let size = max(0, fileSizeBytes)

        if size >= heavyFileBytes {
            return PDFProcessingPlan(
                workerCount: min(workers, 2),
                inFlightPerWorker: 2,
                pageBatchSize: 8,
                fileSizeBytes: size)
        }
        if size >= mediumFileBytes {
            return PDFProcessingPlan(
                workerCount: min(workers, 3),
                inFlightPerWorker: 3,
                pageBatchSize: 24,
                fileSizeBytes: size)
        }
        if size >= largeFileBytes {
            return PDFProcessingPlan(
                workerCount: workers,
                inFlightPerWorker: 3,
                pageBatchSize: 32,
                fileSizeBytes: size)
        }
        return PDFProcessingPlan(
            workerCount: workers,
            inFlightPerWorker: 4,
            pageBatchSize: 50,
            fileSizeBytes: size)
    }

    static func plan(fileURL: URL,
                     renderWorkers: Int = PDFPageRenderer.renderWorkers) -> PDFProcessingPlan {
        plan(fileSizeBytes: fileSize(of: fileURL), renderWorkers: renderWorkers)
    }

    /// 读取文件体积；路径不可读时按 0 处理（保持默认并发）。
    static func fileSize(of url: URL) -> Int64 {
        if let values = try? url.resourceValues(forKeys: [.fileSizeKey]),
           let size = values.fileSize {
            return Int64(max(0, size))
        }
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        return (attributes?[.size] as? NSNumber)?.int64Value ?? 0
    }
}

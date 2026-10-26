import Foundation
import CoreGraphics

/// 扫描页统一执行器：多 worker 并行产图 + 渲染/OCR 流水线重叠。
///
/// 结构（对应渲染优化的 ①③④ 三项）：
/// - 每 worker 独占一个 `CGPDFDocument` 实例，只在自己的任务里访问，
///   规避 PDFPage/CGPDFDocument 的线程安全问题，无需加锁；
/// - worker 内串行产图（[PDFPageRenderer.produceImage]：优先直接提取内嵌
///   JPEG，失败回退 CoreGraphics 栅格化），每产出一页立即把 OCR 提交到
///   内层任务组，不等后续页渲染——渲染与识别跨页重叠；
/// - worker 之间并行（数量 [PDFPageRenderer.renderWorkers]），
///   多文件同时解析时总计渲染并发由该值间接约束。
///
/// OCR 并发不受本执行器管理：Vision 走全局 visionGate，Ovis 走实例池，
/// 由调用方在 `perPage` 闭包里自行限流。
enum PDFPageScanRunner {

    struct Outcome {
        let index: Int
        /// 空白页 / 产图失败为 true：调用方按空文本处理。
        let isBlank: Bool
        /// perPage 产出的文本；nil 表示该页识别失败（或需调用方另行处理）。
        let text: String?
        let needsVLM: Bool
        /// 每页耗时（秒），供 metadata 统计（worker 间并行，累计值是各阶段工作量）。
        var renderSeconds: Double = 0
        var ocrSeconds: Double = 0
        /// 识别完成即释放图像；Outcome 一律不携带 CGImage——需要图像的调用方
        /// （如 Ovis 兜底）按 index 重新产图。否则全书页面图像会随 outcomes
        /// 数组驻留到执行器结束，多文件并发时内存失控。
    }

    /// 每 worker 同时挂起的识别任务上限。产图（11-26ms/页）远快于识别
    /// （0.5-数秒/页），无窗口时全部页面图像会积压在挂起任务闭包里。
    /// 窗口给产图加了背压：渲染与识别仍跨页重叠，但驻留图像数量有界。
    private static let inFlightLimit = 4

    /// 分批大小：一次只把这么多页交给流水线，批处理完再给下一批。
    /// 配合文件级并发（默认 6），整库图像在途上限 ≈ 50 × 并发文件数；
    /// 超大 PDF（数千页）不会再把全书页面一次性压进内存。
    private static let pageBatchSize = 50

    /// 执行扫描页产图 + OCR。
    ///
    /// - Parameters:
    ///   - pages: 需要视觉处理的 0-based 页序。
    ///   - targetLongSide: 渲染/提取目标长边（Vision 1500 / Ovis 1408）。
    ///   - dpi: 内嵌图提取失败回退栅格化时的 DPI。
    ///   - perPage: 对单页图像执行识别；返回的 `text == nil` 时该页记为
    ///     需要调用方兜底（`needsVLM` 标记原因）；`busySeconds` 为识别
    ///     实际占用时长（不含排队等待），用于 metadata 统计。
    static func run(url: URL,
                    pages: [Int],
                    pageTotal: Int,
                    targetLongSide: Int,
                    dpi: Double,
                    progress: ((ParserProgress) -> Void)?,
                    perPage: @escaping (_ pageIndex: Int, _ image: CGImage) async throws
                        -> (text: String?, needsVLM: Bool, busySeconds: Double)) async throws -> [Outcome] {
        guard !pages.isEmpty else { return [] }

        // CGPDFDocument 实例在 worker 任务里独占使用；批间串行（前一批的
        // 任务组完全结束后才开始下一批），所以跨批复用是安全的。提前释放引用。
        var documents: [CGPDFDocument] = []
        documents.reserveCapacity(PDFPageRenderer.renderWorkers)
        for _ in 0..<PDFPageRenderer.renderWorkers {
            guard let doc = CGPDFDocument(url as CFURL) else {
                throw AgentError.parse("无法打开 PDF：\(url.lastPathComponent)")
            }
            if doc.isEncrypted, !doc.unlockWithPassword("") {
                throw AgentError.parse("PDF 已加密且无法用空密码解锁：\(url.lastPathComponent)")
            }
            documents.append(doc)
        }
        defer { documents.removeAll() }

        // 分批执行：每批 pageBatchSize 页，批间串行——前一批全部识别完成后
        // 才开始产下一批的图，超大 PDF 的在途图像因此有硬上限。
        var all: [Outcome] = []
        for offset in stride(from: 0, to: pages.count, by: pageBatchSize) {
            try Task.checkCancellation()
            let batch = Array(pages[offset..<min(offset + pageBatchSize, pages.count)])
            all.append(contentsOf: try await runBatch(
                batch, documents: documents, pageTotal: pageTotal,
                targetLongSide: targetLongSide, dpi: dpi,
                progress: progress, perPage: perPage))
        }
        return all.sorted { $0.index < $1.index }
    }

    /// 执行一批页面的产图 + OCR（页数 ≤ pageBatchSize）。
    private static func runBatch(_ pages: [Int],
                                 documents: [CGPDFDocument],
                                 pageTotal: Int,
                                 targetLongSide: Int,
                                 dpi: Double,
                                 progress: ((ParserProgress) -> Void)?,
                                 perPage: @escaping (_ pageIndex: Int, _ image: CGImage) async throws
                                     -> (text: String?, needsVLM: Bool, busySeconds: Double)) async throws -> [Outcome] {
        let workerCount = max(1, min(documents.count, pages.count))

        return try await withThrowingTaskGroup(of: [Outcome].self) { group in
            for worker in 0..<workerCount {
                let myPages = pages.enumerated()
                    .filter { $0.offset % workerCount == worker }
                    .map(\.element)
                let document = documents[worker]
                group.addTask {
                    var outcomes: [Outcome] = []
                    try await withThrowingTaskGroup(of: Outcome.self) { inner in
                        var pagesIter = myPages.makeIterator()

                        // 产下一页并提交识别；返回 false 表示没有更多页。
                        func submitNext() throws -> Bool {
                            guard let pageIndex = pagesIter.next() else { return false }
                            try Task.checkCancellation()
                            progress?(.page(index: pageIndex + 1,
                                            total: pageTotal, stage: "render"))
                            let renderStarted = Date()
                            let image = PDFPageRenderer.produceImage(
                                document: document,
                                pageIndex: pageIndex,
                                targetLongSide: targetLongSide,
                                dpi: dpi)
                            let renderSeconds = Date().timeIntervalSince(renderStarted)
                            guard let image,
                                  !PDFPageRenderer.isNearlyBlank(image) else {
                                outcomes.append(Outcome(index: pageIndex, isBlank: true,
                                                        text: nil, needsVLM: false,
                                                        renderSeconds: renderSeconds))
                                return true
                            }
                            inner.addTask {
                                let result = try await perPage(pageIndex, image)
                                return Outcome(index: pageIndex, isBlank: false,
                                               text: result.text, needsVLM: result.needsVLM,
                                               renderSeconds: renderSeconds,
                                               ocrSeconds: result.busySeconds)
                            }
                            return true
                        }

                        // 先填满在途窗口，之后每完成一页补一页。
                        var inFlight = 0
                        while inFlight < inFlightLimit, try submitNext() {
                            inFlight += 1
                        }
                        while let outcome = try await inner.next() {
                            outcomes.append(outcome)
                            inFlight -= 1
                            while inFlight < inFlightLimit, try submitNext() {
                                inFlight += 1
                            }
                        }
                    }
                    return outcomes
                }
            }

            var all: [Outcome] = []
            for try await chunk in group {
                all.append(contentsOf: chunk)
            }
            return all.sorted { $0.index < $1.index }
        }
    }
}

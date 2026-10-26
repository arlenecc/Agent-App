import Foundation
import Darwin

/// 异步互斥锁：让同一时刻只有一个 embedding/OCR 调用进入模型，
/// 避免多个文件并发时争抢同一个 MLX/ONNX 实例。
final class AsyncGate: @unchecked Sendable {
    private let lock = NSLock()
    private var acquired = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        lock.lock()
        if !acquired {
            acquired = true
            lock.unlock()
            return
        }
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
            lock.unlock()
        }
    }

    func release() {
        lock.lock()
        if let waiter = waiters.first {
            waiters.removeFirst()
            lock.unlock()
            waiter.resume()
        } else {
            acquired = false
            lock.unlock()
        }
    }

    func withLock<T>(_ body: () async throws -> T) async rethrows -> T {
        await acquire()
        do {
            let value = try await body()
            release()
            return value
        } catch {
            release()
            throw error
        }
    }
}

/// 计数信号量：用于限制跨文件的全局 CPU 密集任务并发
/// （例如 Apple Vision OCR），避免文件级并发叠加后过度争抢核心。
/// 等待许可时响应任务取消，避免取消后的同步仍被挂在这里。
final class AsyncSemaphore: @unchecked Sendable {
    private struct Waiter {
        let id: UInt64
        let continuation: CheckedContinuation<Bool, Never>
    }

    private let lock = NSLock()
    private var available: Int
    private var waiters: [Waiter] = []
    private var nextID: UInt64 = 0

    init(value: Int) {
        self.available = max(1, value)
    }

    /// 返回 false 表示等待期间任务被取消，未获得许可（不应 release）。
    func acquire() async -> Bool {
        lock.lock()
        if Task.isCancelled {
            lock.unlock()
            return false
        }
        if available > 0 {
            available -= 1
            lock.unlock()
            return true
        }
        let id = nextID
        nextID &+= 1
        lock.unlock()

        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                lock.lock()
                if Task.isCancelled {
                    lock.unlock()
                    continuation.resume(returning: false)
                    return
                }
                waiters.append(Waiter(id: id, continuation: continuation))
                lock.unlock()
            }
        } onCancel: {
            lock.lock()
            if let index = waiters.firstIndex(where: { $0.id == id }) {
                let waiter = waiters.remove(at: index)
                lock.unlock()
                waiter.continuation.resume(returning: false)
            } else {
                lock.unlock()
            }
        }
    }

    func release() {
        lock.lock()
        if let waiter = waiters.first {
            waiters.removeFirst()
            lock.unlock()
            waiter.continuation.resume(returning: true)
        } else {
            available += 1
            lock.unlock()
        }
    }

    func withPermit<T>(_ body: () async throws -> T) async throws -> T {
        guard await acquire() else { throw AgentError.cancelled }
        // 许可刚被 release 唤醒时任务可能已被取消：直接归还并退出。
        if Task.isCancelled {
            release()
            throw AgentError.cancelled
        }
        defer { release() }
        return try await body()
    }
}


/// 进程当前物理内存占用（包含 MLX 权重、KV/buffer 与 ONNX 张量）。
enum ProcessMemory {
    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / 1_048_576
    }
}

/// 多文件并发 ingest 时共享的 embedding 调度器：
/// - 按底层 provider 的并发能力限流（MLX 串行，ONNX 可 2 路并发）；
/// - 按吞吐与内存自动调整子批大小（默认 2...64，初始 16，步进 2）；
/// - 内存超过上限时立即下调，吞吐下降时反向探测，持续寻找当前内存下的最优值。
public final class AdaptiveEmbeddingScheduler: EmbeddingProvider, @unchecked Sendable {

    public struct Options: Sendable {
        public var minBatchSize: Int
        public var maxBatchSize: Int
        public var initialBatchSize: Int
        public var step: Int
        /// 子批调参的内存上限（MB），默认 6GB。
        public var memoryLimitMB: Double
        /// 每个尺寸至少观察的批次数或秒数，避免测量噪声导致抖动。
        public var probeBatches: Int
        public var probeSeconds: Double

        public init(minBatchSize: Int = 2,
                    maxBatchSize: Int = 64,
                    initialBatchSize: Int = 16,
                    step: Int = 2,
                    memoryLimitMB: Double = 6 * 1_024,
                    probeBatches: Int = 2,
                    probeSeconds: Double = 2.0) {
            self.minBatchSize = minBatchSize
            self.maxBatchSize = maxBatchSize
            self.initialBatchSize = initialBatchSize
            self.step = step
            self.memoryLimitMB = memoryLimitMB
            self.probeBatches = probeBatches
            self.probeSeconds = probeSeconds
        }
    }

    private let base: EmbeddingProvider
    private let options: Options
    private let log: (String) -> Void
    /// 并发闸门：按底层模型的并发能力限流（ONNX 可 2 路并发 run、
    /// MLX 串行）。信号量在等待时响应取消（AsyncGate 不会）。
    private let semaphore: AsyncSemaphore
    private let stateLock = NSLock()

    // stateLock 保护以下状态
    private var batchSize: Int
    private var windowChunks = 0
    private var windowSeconds = 0.0
    private var windowBatches = 0
    private var windowStarted = Date()
    private var lastThroughput = 0.0
    private var direction = 1   // +1 上探，-1 下探
    private var lastAdjustmentLog = Date.distantPast

    public var identifier: String { base.identifier }
    public var dimension: Int { base.dimension }

    /// 供完成日志展示当前子批大小。
    public var currentBatchSize: Int {
        stateLock.lock()
        defer { stateLock.unlock() }
        return batchSize
    }

    public init(base: EmbeddingProvider,
                options: Options = Options(),
                log: @escaping (String) -> Void = { AgentLog.info($0) }) {
        self.base = base
        self.options = options
        self.log = log
        self.semaphore = AsyncSemaphore(value: max(1, base.maxConcurrentRuns))
        self.batchSize = max(options.minBatchSize,
                             min(options.initialBatchSize, options.maxBatchSize))
    }

    // MARK: - EmbeddingProvider

    public func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        try await embedDocuments(texts, progress: nil)
    }

    /// 用于流水线内的长文档：每完成一个子批回调一次进度。
    public func embedDocuments(
        _ texts: [String],
        progress: ((Int, Int) -> Void)?
    ) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        return try await semaphore.withPermit {
            var result: [[Float]] = []
            result.reserveCapacity(texts.count)
            var offset = 0
            while offset < texts.count {
                if Task.isCancelled { throw AgentError.cancelled }
                let size = currentBatchSize
                let end = min(offset + size, texts.count)
                let batch = Array(texts[offset..<end])
                let batchStarted = Date()
                let vectors = try await base.embedDocuments(batch)
                let seconds = Date().timeIntervalSince(batchStarted)
                guard vectors.count == batch.count else {
                    throw AgentError.storage(
                        "embedding 子批返回数量异常: \(vectors.count) != \(batch.count)")
                }
                result.append(contentsOf: vectors)
                offset = end
                record(chunks: batch.count, seconds: seconds)
                progress?(offset, texts.count)
            }
            return result
        }
    }

    public func embedQuery(_ text: String) async throws -> [Float] {
        try await semaphore.withPermit {
            try await base.embedQuery(text)
        }
    }

    /// 转发到底层 provider 释放权重。
    public func unload() async {
        await base.unload()
    }

    // MARK: - 自适应子批

    private func record(chunks: Int, seconds: TimeInterval) {
        guard seconds > 0.0001 else { return }

        stateLock.lock()
        windowChunks += chunks
        windowSeconds += seconds
        windowBatches += 1
        let memoryMB = ProcessMemory.footprintMB()
        let enoughBatches = windowBatches >= max(1, options.probeBatches)
        let enoughTime = Date().timeIntervalSince(windowStarted) >= options.probeSeconds
        guard enoughBatches || enoughTime else {
            stateLock.unlock()
            return
        }

        let throughput = Double(windowChunks) / max(windowSeconds, 0.0001)
        let oldSize = batchSize
        var newSize = batchSize
        var reason: String?

        if memoryMB > options.memoryLimitMB {
            if batchSize > options.minBatchSize {
                direction = -1
                newSize = max(options.minBatchSize, batchSize - options.step)
                reason = "内存 \(Int(memoryMB))MB 超限"
            }
        } else if lastThroughput <= 0 {
            if batchSize < options.maxBatchSize {
                direction = 1
                newSize = min(options.maxBatchSize, batchSize + options.step)
                reason = "首次上探"
            }
        } else {
            let ratio = throughput / lastThroughput
            if ratio >= 1.03 {
                if direction > 0, batchSize < options.maxBatchSize {
                    newSize = min(options.maxBatchSize, batchSize + options.step)
                    reason = "吞吐提升，继续上探"
                } else if direction < 0, batchSize > options.minBatchSize {
                    newSize = max(options.minBatchSize, batchSize - options.step)
                    reason = "下调后吞吐提升，继续下探"
                }
            } else if ratio <= 0.97 {
                direction = -direction
                if direction > 0, batchSize < options.maxBatchSize {
                    newSize = min(options.maxBatchSize, batchSize + options.step)
                    reason = "下探无提升，回升"
                } else if direction < 0, batchSize > options.minBatchSize {
                    newSize = max(options.minBatchSize, batchSize - options.step)
                    reason = "吞吐下降，下调"
                }
            }
        }

        lastThroughput = throughput
        windowChunks = 0
        windowSeconds = 0
        windowBatches = 0
        windowStarted = Date()
        batchSize = newSize
        stateLock.unlock()

        guard newSize != oldSize, let reason else { return }
        // 调参日志节流，避免快速上探时刷屏。
        guard Date().timeIntervalSince(lastAdjustmentLog) >= 10 else { return }
        lastAdjustmentLog = Date()
        log("RAG: embedding 子批自适应 \(oldSize) → \(newSize)"
            + "（\(reason)，\(Int(throughput)) 片/秒，内存 \(Int(memoryMB))/\(Int(options.memoryLimitMB)) MB）")
    }
}

import Foundation
import OnnxRuntimeBindings

/// ONNX Runtime 封装层错误。
public enum ONNXRuntimeError: LocalizedError, Sendable {
    case environment(String)
    case session(String)
    case value(String)
    case input(String)
    case output(String)
    case shape(String)

    public var errorDescription: String? {
        switch self {
        case .environment(let message): return "ONNX Runtime 环境初始化失败：\(message)"
        case .session(let message): return "ONNX Runtime 会话创建失败：\(message)"
        case .value(let message): return "ONNX Runtime 张量操作失败：\(message)"
        case .input(let message): return "ONNX Runtime 输入不合法：\(message)"
        case .output(let message): return "ONNX Runtime 输出不合法：\(message)"
        case .shape(let message): return "ONNX Runtime 张量维度不合法：\(message)"
        }
    }
}

/// 浮点张量（ONNX 常见输出）。
public struct OrtTensor: Sendable {
    public var values: [Float]
    /// 运行时实际维度；输入时动态维度由调用方给出的 shape 决定。
    public var shape: [Int]

    public init(values: [Float], shape: [Int]) {
        self.values = values
        self.shape = shape
    }
}

/// int64 张量（token ids）。
public struct OrtInt64Tensor: Sendable {
    public var values: [Int64]
    public var shape: [Int]

    public init(values: [Int64], shape: [Int]) {
        self.values = values
        self.shape = shape
    }
}

/// 全局 ORT 环境。进程内只创建一次。
public final class OrtEnv: @unchecked Sendable {
    private static let cacheLock = NSLock()
    nonisolated(unsafe) private static var cached: OrtEnv?

    /// 获取进程级共享环境。
    public static func shared() throws -> OrtEnv {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        if let cached { return cached }
        let instance = try OrtEnv()
        cached = instance
        return instance
    }

    let raw: ORTEnv

    public init(loggingLevel: ORTLoggingLevel = .warning) throws {
        do {
            self.raw = try ORTEnv(loggingLevel: loggingLevel)
        } catch {
            throw ONNXRuntimeError.environment(error.localizedDescription)
        }
    }
}

/// ONNX 会话。ORT 的 `Run` 本身是线程安全的，因此包装类可安全跨 actor 使用。
public final class OrtSession: @unchecked Sendable {
    private let session: ORTSession
    public let modelPath: String
    /// 模型声明的输入/输出名（顺序来自 ONNX 图）。
    public let inputNames: [String]
    public let outputNames: [String]

    public init(
        modelPath: String,
        environment: OrtEnv? = nil,
        intraOpThreads: Int = 4,
        optimization: ORTGraphOptimizationLevel = .all
    ) throws {
        let env: OrtEnv
        if let environment {
            env = environment
        } else {
            env = try OrtEnv.shared()
        }
        self.modelPath = modelPath

        do {
            let options = try ORTSessionOptions()
            if intraOpThreads > 0 {
                try options.setIntraOpNumThreads(Int32(intraOpThreads))
            }
            try options.setGraphOptimizationLevel(optimization)

            self.session = try ORTSession(
                env: env.raw,
                modelPath: modelPath,
                sessionOptions: options
            )
        } catch {
            throw ONNXRuntimeError.session("\(error.localizedDescription)（模型：\(modelPath)）")
        }

        do {
            let inputs = try session.inputNames()
            guard !inputs.isEmpty else {
                throw ONNXRuntimeError.session("模型没有声明输入：\(modelPath)")
            }
            let outputs = try session.outputNames()
            guard !outputs.isEmpty else {
                throw ONNXRuntimeError.session("模型没有声明输出：\(modelPath)")
            }
            self.inputNames = inputs
            self.outputNames = outputs
        } catch let error as ONNXRuntimeError {
            throw error
        } catch {
            throw ONNXRuntimeError.session("读取模型输入输出名失败：\(error.localizedDescription)")
        }
    }

    /// 执行推理。动态维度由输入张量的 shape 决定，输出维度以运行时结果为准。
    public func run(
        inputs: [String: OrtValue],
        outputNames requested: [String]? = nil
    ) throws -> [String: OrtValue] {
        let rawInputs = inputs.mapValues { $0.raw }
        let names = requested ?? outputNames

        do {
            let outputs = try session.run(
                withInputs: rawInputs,
                outputNames: Set(names),
                runOptions: nil
            )
            return outputs.mapValues { OrtValue(wrapping: $0) }
        } catch let error as ONNXRuntimeError {
            throw error
        } catch {
            throw ONNXRuntimeError.output("\(error.localizedDescription)（模型：\(modelPath)）")
        }
    }
}

/// ORT 张量值。创建输入时持有底层 NSMutableData，避免 runtime 读到已释放内存。
public final class OrtValue: @unchecked Sendable {
    let raw: ORTValue
    /// 输入张量由调用方分配内存；ORT 不保证延长其生命周期。
    private let storage: NSMutableData?

    init(wrapping raw: ORTValue) {
        self.raw = raw
        self.storage = nil
    }

    public init(float32 values: [Float], shape: [Int]) throws {
        let data: NSMutableData
        if values.isEmpty {
            data = NSMutableData()
        } else {
            data = values.withUnsafeBufferPointer { buffer in
                NSMutableData(bytes: buffer.baseAddress!, length: buffer.count * MemoryLayout<Float>.stride)
            }
        }
        self.storage = data
        do {
            self.raw = try ORTValue(
                tensorData: data,
                elementType: .float,
                shape: shape.map { NSNumber(value: $0) }
            )
        } catch {
            throw ONNXRuntimeError.value(error.localizedDescription)
        }
    }

    public init(int64 values: [Int64], shape: [Int]) throws {
        let data: NSMutableData
        if values.isEmpty {
            data = NSMutableData()
        } else {
            data = values.withUnsafeBufferPointer { buffer in
                NSMutableData(bytes: buffer.baseAddress!, length: buffer.count * MemoryLayout<Int64>.stride)
            }
        }
        self.storage = data
        do {
            self.raw = try ORTValue(
                tensorData: data,
                elementType: .int64,
                shape: shape.map { NSNumber(value: $0) }
            )
        } catch {
            throw ONNXRuntimeError.value(error.localizedDescription)
        }
    }

    /// 读取 float32 张量。
    public func floatTensor() throws -> OrtTensor {
        let info = try tensorInfo()
        guard info.elementType == .float else {
            throw ONNXRuntimeError.output("期望 float32 输出，实际为 \(info.elementType.rawValue)")
        }
        let shape = info.shape.map { $0.intValue }
        let data: NSMutableData
        do {
            data = try raw.tensorData()
        } catch {
            throw ONNXRuntimeError.output(error.localizedDescription)
        }
        let count = data.length / MemoryLayout<Float>.stride
        var values = [Float](repeating: 0, count: count)
        if count > 0 {
            values.withUnsafeMutableBytes { buffer in
                data.getBytes(buffer.baseAddress!, length: data.length)
            }
        }
        return OrtTensor(values: values, shape: shape)
    }

    /// 读取 int64 张量（部分导出模型会输出 token ids / position_ids）。
    public func int64Tensor() throws -> OrtInt64Tensor {
        let info = try tensorInfo()
        guard info.elementType == .int64 else {
            throw ONNXRuntimeError.output("期望 int64 输出，实际为 \(info.elementType.rawValue)")
        }
        let shape = info.shape.map { $0.intValue }
        let data: NSMutableData
        do {
            data = try raw.tensorData()
        } catch {
            throw ONNXRuntimeError.output(error.localizedDescription)
        }
        let count = data.length / MemoryLayout<Int64>.stride
        var values = [Int64](repeating: 0, count: count)
        if count > 0 {
            values.withUnsafeMutableBytes { buffer in
                data.getBytes(buffer.baseAddress!, length: data.length)
            }
        }
        return OrtInt64Tensor(values: values, shape: shape)
    }

    private func tensorInfo() throws -> ORTTensorTypeAndShapeInfo {
        do {
            return try raw.tensorTypeAndShapeInfo()
        } catch {
            throw ONNXRuntimeError.output(error.localizedDescription)
        }
    }
}

import Foundation

/// 向量落盘元素类型。对应 sqlite-vec 的三种原生列类型：
/// float32（`float[N]`）、int8（`int8[N]`）、bit（`bit[N]`）。
///
/// `fp16` 保留为配置项以便与其它向量库对齐，但 sqlite-vec 0.1.9 没有
/// fp16 元素类型，选择它会在打开向量库时直接报错并提示改用其它类型。
public enum VectorElementType: String, CaseIterable, Sendable {
    case fp32
    case int8
    case fp16
    case bin

    /// 默认元素类型：int8（默认模型截断到 256 维，每行 256B，是 fp32 的 1/4）。
    public static let defaultValue: VectorElementType = .int8

    public var isSupported: Bool { self != .fp16 }

    /// 不支持时的明确原因（状态面板与设置校验共用同一句提示）。
    public var unsupportedReason: String? {
        guard !isSupported else { return nil }
        return "sqlite-vec 0.1.9 不支持 fp16 向量；请把 rag_vector_element_type "
            + "改为 int8（推荐）/ fp32 / bin"
    }

    public var displayName: String { rawValue }

    /// 设置面板/状态栏用的一句话说明。
    public var summary: String {
        switch self {
        case .fp32: return "fp32：4 字节/维，余弦距离，兼容旧库"
        case .int8: return "int8：1 字节/维，余弦距离（默认）"
        case .fp16: return "fp16：sqlite-vec 不支持，需改用 int8 / fp32 / bin"
        case .bin: return "bin：1 bit/维，汉明距离，体积最小"
        }
    }

    /// bit 列只能是汉明距离（vec0 显式拒绝 `distance_metric`）。
    public var isBinary: Bool { self == .bin }

    var sqlColumnType: String {
        switch self {
        case .fp32, .fp16: return "float"
        case .int8: return "int8"
        case .bin: return "bit"
        }
    }

    /// vec0 建表列定义。
    func columnDefinition(dimension: Int) -> String {
        if isBinary {
            // bit 列不接受 distance_metric，vec0 固定用汉明距离。
            return "embedding bit[\(dimension)]"
        }
        return "embedding \(sqlColumnType)[\(dimension)] distance_metric=cosine"
    }

    /// 读取 sqlite_master 里的建表语句后判断列类型。
    static func infer(fromSchema sql: String) -> VectorElementType? {
        let lower = sql.lowercased()
        guard lower.contains("using vec0") else { return nil }
        // 先看 bit，避免 "bit[" 里的 "int8" 之类误判（目前不会同时出现）。
        if lower.contains(" bit[") || lower.contains("(bit[") { return .bin }
        if lower.contains(" int8[") { return .int8 }
        if lower.contains(" float[") { return .fp32 }
        return nil
    }

    /// 写入 / KNN 的绑定表达式：int8 与 bit 必须包一层，让 sqlite-vec
    /// 按 subtype 识别元素类型（裸 BLOB 一律按 float32 解析）。
    func bindExpression(_ placeholder: String) -> String {
        switch self {
        case .int8: return "vec_int8(\(placeholder))"
        case .bin: return "vec_bit(\(placeholder))"
        case .fp32, .fp16: return placeholder
        }
    }

    /// 单行落盘字节数。
    func blobSize(dimension: Int) -> Int {
        switch self {
        case .fp32, .fp16: return dimension * 4
        case .int8: return dimension
        case .bin: return (dimension + 7) / 8
        }
    }

    /// sqlite-vec 返回的 distance → 相似度（越大越相关）。
    func similarity(distance: Double, dimension: Int) -> Double {
        guard isBinary else { return max(0, 1 - distance) }
        guard dimension > 0 else { return 0 }
        return max(0, 1 - distance / Double(dimension))
    }

    /// 建表前的合法性校验。
    func validate(dimension: Int) throws {
        if let unsupportedReason {
            throw AgentError.storage(unsupportedReason)
        }
        guard dimension > 0 else {
            throw AgentError.storage("向量维度无效: \(dimension)")
        }
        if isBinary, dimension % 8 != 0 {
            throw AgentError.storage(
                "bin 元素类型要求维度是 8 的倍数，当前维度 \(dimension)")
        }
    }
}

// MARK: - 编解码

/// 向量与落盘字节之间的唯一转换入口。
///
/// 入库与查询都必须经过这里：`int8`/`bin` 的量化是有损的，两侧
/// 用同一套规则才能保证余弦/汉明打分一致。
public enum VectorEncoder {
    /// 编码为落盘字节。**入参必须是已归一化的向量**（int8/bin 依赖 |x| ≤ 1）。
    public static func encode(_ vector: [Float], as type: VectorElementType) -> Data {
        switch type {
        case .fp32, .fp16:
            return VectorCodec.encode(vector)
        case .int8:
            return encodeInt8(vector)
        case .bin:
            return encodeBits(vector)
        }
    }

    /// 落盘字节还原为近似向量（fp32 原样、int8 ÷127、bin ±1/√dim）。
    public static func decode(_ data: Data, as type: VectorElementType,
                              dimension: Int) -> [Float] {
        switch type {
        case .fp32, .fp16:
            return VectorCodec.decode(data)
        case .int8:
            guard dimension > 0, data.count >= dimension else { return [] }
            return dequantizeInt8(data, dimension: dimension)
        case .bin:
            guard dimension > 0, data.count * 8 >= dimension else { return [] }
            return dequantizeBits(data, dimension: dimension)
        }
    }

    // MARK: int8

    /// 对称量化：`q = clamp(round(x × 127), -127, 127)`。
    /// 不做 `[-128,127]` 的非对称映射——那会引入常数偏移，破坏余弦。
    public static func encodeInt8(_ vector: [Float]) -> Data {
        var bytes = [UInt8](repeating: 0, count: vector.count)
        for (index, value) in vector.enumerated() {
            let quantized = (value * 127).rounded()
            bytes[index] = UInt8(bitPattern: Int8(max(-127, min(127, quantized))))
        }
        return Data(bytes)
    }

    public static func dequantizeInt8(_ data: Data, dimension: Int) -> [Float] {
        var output = [Float](repeating: 0, count: dimension)
        data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            let codes = base.assumingMemoryBound(to: Int8.self)
            for i in 0..<dimension { output[i] = Float(codes[i]) / 127 }
        }
        return output
    }

    // MARK: bin

    /// 符号位量化：`bit = x > 0`，低位在前（与 sqlite-vec 的
    /// `vec_quantize_binary` 一致，两者产出的 blob 可以互相比较）。
    public static func encodeBits(_ vector: [Float]) -> Data {
        let byteCount = max(1, (vector.count + 7) / 8)
        var bytes = [UInt8](repeating: 0, count: byteCount)
        for (index, value) in vector.enumerated() where value > 0 {
            bytes[index / 8] |= UInt8(1 << (index % 8))
        }
        return Data(bytes)
    }

    /// 位向量还原为 ±1/√dim：这样每行 L2 范数为 1，余弦与汉明排序同向。
    public static func dequantizeBits(_ data: Data, dimension: Int) -> [Float] {
        var output = [Float](repeating: 0, count: dimension)
        let magnitude = dimension > 0 ? 1 / Float(dimension).squareRoot() : 0
        data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            let bits = base.assumingMemoryBound(to: UInt8.self)
            for i in 0..<dimension {
                let byte = i / 8
                guard byte < raw.count else { break }
                let on = (bits[byte] >> UInt8(i % 8)) & 1 == 1
                output[i] = on ? magnitude : -magnitude
            }
        }
        return output
    }

    /// 位向量的汉明距离（供精确扫描打分）。
    public static func hamming(_ a: Data, _ b: Data) -> Int {
        let count = min(a.count, b.count)
        guard count > 0 else { return 0 }
        return a.withUnsafeBytes { left in
            b.withUnsafeBytes { right in
                guard let l = left.baseAddress, let r = right.baseAddress else { return 0 }
                let lp = l.assumingMemoryBound(to: UInt8.self)
                let rp = r.assumingMemoryBound(to: UInt8.self)
                var distance = 0
                for i in 0..<count { distance += (lp[i] ^ rp[i]).nonzeroBitCount }
                return distance
            }
        }
    }

    /// 内存指针与位图数组的汉明距离（ANN 热路径：不复制候选行）。
    public static func hamming(pointer: UnsafeRawPointer, bytes: [UInt8],
                               count: Int) -> Int {
        guard count > 0, bytes.count >= count else { return 0 }
        return bytes.withUnsafeBytes { reference -> Int in
            guard let referenceBase = reference.baseAddress else { return 0 }
            let left = referenceBase.assumingMemoryBound(to: UInt8.self)
            let right = pointer.assumingMemoryBound(to: UInt8.self)
            var distance = 0
            for i in 0..<count { distance += (left[i] ^ right[i]).nonzeroBitCount }
            return distance
        }
    }
}

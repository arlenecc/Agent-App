import XCTest
@testable import BaseAgentCore

/// 向量落盘类型与编解码：int8 对称量化、bin 位图、fp16 拒绝。
final class VectorElementTypeTests: XCTestCase {
    func testInt8RoundTripPreservesDirection() {
        let vector = VectorCodec.normalize([0.6, 0.8, -0.2, 0.1])
        let data = VectorEncoder.encode(vector, as: .int8)
        XCTAssertEqual(data.count, 4, "int8 每维 1 字节")

        let decoded = VectorEncoder.decode(data, as: .int8, dimension: 4)
        XCTAssertEqual(decoded.count, 4)
        XCTAssertEqual(VectorCodec.cosine(vector, decoded), 1.0, accuracy: 0.001)
        for (original, restored) in zip(vector, decoded) {
            XCTAssertEqual(restored, original, accuracy: 1.0 / 127 + 0.0001)
        }
    }

    /// 必须是 `[-127,127]` 的对称映射：非对称映射会引入常数偏移、破坏余弦。
    func testInt8QuantizationIsSymmetric() {
        let data = VectorEncoder.encode([2, -2, 0.5, 0], as: .int8)
        let codes = [UInt8](data).map { Int8(bitPattern: $0) }
        XCTAssertEqual(codes, [127, -127, 64, 0])
    }

    func testBinaryRoundTripAndHammingSimilarity() {
        // LSB-first：第 0 位对应第 0 维。
        let vector: [Float] = [1, -1, 1, -1, 1, -1, 1, -1]
        let data = VectorEncoder.encode(vector, as: .bin)
        XCTAssertEqual(data, Data([0b01010101]))

        let decoded = VectorEncoder.decode(data, as: .bin, dimension: 8)
        let norm = decoded.reduce(0) { $0 + $1 * $1 }.squareRoot()
        XCTAssertEqual(norm, 1, accuracy: 1e-5, "位向量还原后必须是单位向量")

        XCTAssertEqual(VectorEncoder.hamming(data, data), 0)
        XCTAssertEqual(VectorEncoder.hamming(data, Data([0b10101010])), 8)
        XCTAssertEqual(VectorElementType.bin.similarity(distance: 2, dimension: 8),
                       0.75, accuracy: 1e-9)
    }

    func testFP16IsRejectedWithActionableMessage() {
        XCTAssertThrowsError(try VectorElementType.fp16.validate(dimension: 512)) { error in
            XCTAssertTrue(error.localizedDescription.contains("fp16"))
            XCTAssertTrue(error.localizedDescription.contains("int8"))
        }
        XCTAssertNotNil(VectorElementType.fp16.unsupportedReason)
        XCTAssertNil(VectorElementType.int8.unsupportedReason)
    }

    func testBindExpressionsAndBlobSizes() {
        XCTAssertEqual(VectorElementType.int8.bindExpression("?"), "vec_int8(?)")
        XCTAssertEqual(VectorElementType.bin.bindExpression("?"), "vec_bit(?)")
        XCTAssertEqual(VectorElementType.fp32.bindExpression("?"), "?")

        XCTAssertEqual(VectorElementType.fp32.blobSize(dimension: 512), 2048)
        XCTAssertEqual(VectorElementType.int8.blobSize(dimension: 512), 512)
        XCTAssertEqual(VectorElementType.bin.blobSize(dimension: 512), 64)
    }

    func testSchemaInference() {
        XCTAssertEqual(VectorElementType.infer(
            fromSchema: "CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding int8[512] distance_metric=cosine)"),
            .int8)
        XCTAssertEqual(VectorElementType.infer(
            fromSchema: "CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding bit[512])"),
            .bin)
        XCTAssertEqual(VectorElementType.infer(
            fromSchema: "CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[512] distance_metric=cosine)"),
            .fp32)
    }
}

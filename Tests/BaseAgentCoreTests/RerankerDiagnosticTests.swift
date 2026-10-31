import XCTest
import Foundation
@testable import BaseAgentCore

/// 临时诊断：对比 Swift tokenizer/ONNX 与 Python 参考实现的差异。
///
/// 只打印不断言，且需要真实模型与较长耗时，默认跳过；
/// 排查重排数值问题时用 `RERANKER_DIAG=1 swift test --filter RerankerDiagnosticTests` 运行。
final class RerankerDiagnosticTests: XCTestCase {
    private func skipUnlessEnabled() throws {
        guard ProcessInfo.processInfo.environment["RERANKER_DIAG"] == "1" else {
            throw XCTSkip("临时诊断测试，设置 RERANKER_DIAG=1 运行")
        }
    }

    func testTokenIDsAndScores() async throws {
        try skipUnlessEnabled()
        let env = ProcessInfo.processInfo.environment["EMBED_BENCH_MODELS_DIR"]
            ?? URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("dist/BaseAgent.app/Contents/Resources/models")
                .path
        let directory = URL(fileURLWithPath: env, isDirectory: true)
            .appendingPathComponent("jina-reranker-v2-base-multilingual-onnx")
        guard FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("model_quantized.onnx").path) else {
            throw XCTSkip("模型缺失")
        }

        let targets = EmbeddingBenchmarkTests.targets
        let tokenizer = try await LocalTokenizer.load(directory: directory)
        print("[diag] xlmr=\(tokenizer.usesXLMRPairFormat) pad=\(tokenizer.padTokenId) "
            + "bos=\(tokenizer.base.bosTokenId ?? -1) eos=\(tokenizer.base.eosTokenId ?? -1)")

        let cases: [(String, String, String)] = [
            ("q6-doc06", targets[5].query, targets[5].passage),
            ("q6-doc01", targets[5].query, targets[0].passage),
            ("q4-doc04", targets[3].query, targets[3].passage),
            ("q4-doc01", targets[3].query, targets[0].passage),
        ]
        for (label, query, document) in cases {
            let ids = tokenizer.encodePair(query: query, document: document, maxLength: 1024)
            print("[diag] \(label) count=\(ids.count) first20=\(ids.prefix(20)) "
                + "last20=\(ids.suffix(20))")
            let queryIDs = tokenizer.encode(query, addSpecialTokens: false)
            let documentIDs = tokenizer.encode(document, addSpecialTokens: false)
            print("[diag] \(label) qlen=\(queryIDs.count) qfirst=\(queryIDs.prefix(8)) "
                + "dlen=\(documentIDs.count) dfirst=\(documentIDs.prefix(8)) "
                + "dlast=\(documentIDs.suffix(8))")
        }

        let reranker = ONNXReranker(directory: directory, modelName: "jina-reranker")
        let q6Scores = try await reranker.rerank(
            query: targets[5].query,
            documents: [targets[5].passage, targets[0].passage])
        let q4Scores = try await reranker.rerank(
            query: targets[3].query,
            documents: [targets[3].passage, targets[0].passage])
        print("[diag] swift q6 doc06/doc01 = \(q6Scores)")
        print("[diag] swift q4 doc04/doc01 = \(q4Scores)")

        // 批大小对分数/排序的影响：同一组文档分别用 batchSize 1 / 2 / 8 重排。
        let batch1 = ONNXReranker(directory: directory, modelName: "jina-reranker-b1",
                                  batchSize: 1)
        let batch2 = ONNXReranker(directory: directory, modelName: "jina-reranker-b2",
                                  batchSize: 2)
        for (label, query) in [("q6", targets[5].query), ("q4", targets[3].query)] {
            let s8 = try await reranker.rerank(query: query, documents: targets.map(\.passage))
            let s2 = try await batch2.rerank(query: query, documents: targets.map(\.passage))
            let s1 = try await batch1.rerank(query: query, documents: targets.map(\.passage))
            let fmt = { (v: [Double]) in v.map { String(format: "%.4f", $0) }.joined(separator: ",") }
            print("[diag] \(label) b8=\(fmt(s8))")
            print("[diag] \(label) b2=\(fmt(s2))")
            print("[diag] \(label) b1=\(fmt(s1))")
        }
        await batch1.unload()
        await batch2.unload()

        // 20 条文档的单批 vs 逐条耗时（模拟默认候选池）。
        let manyDocs = Array(repeating: targets.map(\.passage), count: 2).flatMap { $0 }
        let t0 = Date()
        _ = try await reranker.rerank(query: targets[5].query, documents: manyDocs)
        let batchTime = Date().timeIntervalSince(t0)
        let t1 = Date()
        _ = try await batch1.rerank(query: targets[5].query, documents: manyDocs)
        let singleTime = Date().timeIntervalSince(t1)
        print("[diag] latency 20 docs batch8=\(String(format: "%.3f", batchTime))s "
            + "batch1=\(String(format: "%.3f", singleTime))s")
        await reranker.unload()
    }

    /// 隔离 q4-doc04 在 batchSize=1 下分数异常的原因：新会话首条推理 vs 会话历史。
    func testFreshBatchOneIsolation() async throws {
        try skipUnlessEnabled()
        let directory = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("dist/BaseAgent.app/Contents/Resources/models")
            .appendingPathComponent("jina-reranker-v2-base-multilingual-onnx")
        let targets = EmbeddingBenchmarkTests.targets

        let freshSingle = ONNXReranker(directory: directory, modelName: "diag-fresh-b1",
                                       batchSize: 1)
        let onlyDoc = try await freshSingle.rerank(
            query: targets[3].query, documents: [targets[3].passage])
        print("[diag] fresh b1 q4 [doc04] = \(onlyDoc)")
        let pair = try await freshSingle.rerank(
            query: targets[3].query,
            documents: [targets[3].passage, targets[0].passage])
        print("[diag] same-session b1 q4 [doc04,doc01] = \(pair)")
        let ten = try await freshSingle.rerank(
            query: targets[3].query, documents: targets.map(\.passage))
        print("[diag] same-session b1 q4 ten = "
            + ten.map { String(format: "%.4f", $0) }.joined(separator: ","))
        await freshSingle.unload()

        let reversed = try await ONNXReranker(directory: directory, modelName: "diag-rev-b1",
                                              batchSize: 1)
            .rerank(query: targets[3].query,
                    documents: Array(targets.map(\.passage).reversed()))
        print("[diag] fresh b1 q4 reversed = "
            + reversed.map { String(format: "%.4f", $0) }.joined(separator: ","))

        let freshBatch = ONNXReranker(directory: directory, modelName: "diag-fresh-b8",
                                      batchSize: 8)
        let b8Ten = try await freshBatch.rerank(
            query: targets[3].query, documents: targets.map(\.passage))
        print("[diag] fresh b8 q4 ten = "
            + b8Ten.map { String(format: "%.4f", $0) }.joined(separator: ","))
        await freshBatch.unload()
    }

    /// 直接探测同一 pair 在不同 padding 宽度下的 logit，定位 ORT 短序列数值异常。
    func testPaddingWidthProbe() async throws {
        try skipUnlessEnabled()
        let directory = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("dist/BaseAgent.app/Contents/Resources/models")
            .appendingPathComponent("jina-reranker-v2-base-multilingual-onnx")
        let tokenizer = try await LocalTokenizer.load(directory: directory)
        let session = try OrtSession(
            modelPath: directory.appendingPathComponent("model_quantized.onnx").path,
            intraOpThreads: 4)
        let target = EmbeddingBenchmarkTests.targets[3]
        let ids = tokenizer.encodePair(query: target.query, document: target.passage,
                                       maxLength: 1024)
        print("[diag] probe pair length=\(ids.count)")
        for width in [146, 147, 150, 154, 157, 160, 192, 224, 256, 320, 384, 448, 512,
                      640, 768, 896, 926, 1024] {
            guard width >= ids.count else { continue }
            let padded = ids + Array(repeating: tokenizer.padTokenId, count: width - ids.count)
            let mask = Array(repeating: 1, count: ids.count)
                + Array(repeating: 0, count: width - ids.count)
            let inputs = [
                "input_ids": try OrtValue(
                    int64: padded.map(Int64.init), shape: [1, width]),
                "attention_mask": try OrtValue(
                    int64: mask.map(Int64.init), shape: [1, width]),
            ]
            let outputs = try session.run(inputs: inputs, outputNames: ["logits"])
            let logit = try outputs["logits"]!.floatTensor().values.first ?? .nan
            let score = 1 / (1 + exp(-Double(logit)))
            print(String(format: "[diag] width=%d logit=%.4f score=%.4f",
                         width, Double(logit), score))
        }
    }
}

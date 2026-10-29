import XCTest
import Foundation
import MLX
@testable import BaseAgentCore

/// 临时排查用：生产路径（切片 → 调度器 → Matryoshka）与模型裸调的吞吐对比。
/// 设置 `EMBED_PATH_BENCH=1` 启用。
final class EmbeddingPathBenchTests: XCTestCase {
    func testProductionPathThroughput() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["EMBED_PATH_BENCH"] == "1",
            "未设置 EMBED_PATH_BENCH=1")
        let environment = ProcessInfo.processInfo.environment
        let textCount = Int(environment["EMBED_PATH_TEXTS"] ?? "") ?? 16
        let targetTokens = Int(environment["EMBED_PATH_TOKENS"] ?? "") ?? 800

        let modelsRoot = EmbeddingBenchmarkTests.resolveModelsRoot()
        let directory = modelsRoot.appendingPathComponent("jina-embeddings-v3-mlx-int4")
        guard FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("weights.safetensors").path) else {
            throw XCTSkip("jina 模型缺失：\(directory.path)")
        }
        EmbeddingBenchmarkTests.configureMetallib()

        let provider = JinaV3MLXEmbeddingProvider(
            directory: directory, modelName: "jina-embeddings-v3-mlx-int4")
        defer { Task { await provider.unload() } }
        let counterValue = await provider.makeTokenCounter()
        let counter = try XCTUnwrap(counterValue)
        let texts = Self.makeTexts(counter: counter, count: textCount,
                                   targetTokens: targetTokens)
        let tokensPerText = Double(texts.map(counter).reduce(0, +)) / Double(textCount)

        MLX.Memory.cacheLimit = 2_048 * 1_024 * 1_024
        MLX.Memory.clearCache()
        _ = try await provider.embedDocuments(Array(texts.prefix(4)))

        print("[path-bench] texts=\(textCount) tokens/text≈\(String(format: "%.0f", tokensPerText))")

        // 生产路径：Matryoshka(provider) + 调度器默认参数
        let wrapped = MatryoshkaEmbedding.wrap(
            provider, mode: .auto, dimension: 256)
        let scheduler = AdaptiveEmbeddingScheduler(
            base: wrapped,
            options: .init(minBatchSize: 2, maxBatchSize: 32,
                           initialBatchSize: 16, step: 2,
                           memoryLimitMB: 6_144, throughputTolerance: 0.03,
                           probeBatches: 2, probeSeconds: 2.0),
            log: { print("[path-bench][scheduler] \($0)") })

        // 交错两轮：抵消温度/频率/内存压力漂移。
        for round in 0 ..< 2 {
            var start = Date()
            var vectors = try await provider.embedDocuments(texts)
            var seconds = Date().timeIntervalSince(start)
            print(String(format: "[path-bench] r%d 裸调 provider: %.3fs  %.2f chunks/s  %.0f tokens/s  peak %.0fMB",
                         round + 1, seconds, Double(textCount) / seconds,
                         Double(textCount) * tokensPerText / seconds,
                         Double(MLX.Memory.peakMemory) / 1_048_576))
            XCTAssertEqual(vectors.first?.count, 1_024)

            start = Date()
            vectors = try await scheduler.embedDocuments(texts)
            seconds = Date().timeIntervalSince(start)
            print(String(format: "[path-bench] r%d 生产路径 scheduler+MRL: %.3fs  %.2f chunks/s  %.0f tokens/s  batch %d  peak %.0fMB",
                         round + 1, seconds, Double(textCount) / seconds,
                         Double(textCount) * tokensPerText / seconds,
                         scheduler.currentBatchSize,
                         Double(MLX.Memory.peakMemory) / 1_048_576))
            XCTAssertEqual(vectors.first?.count, 256)
        }

        // 3) Matryoshka 截断/归一化本身的成本
        let full = try await provider.embedDocuments(texts)
        let rounds = 200
        var start = Date()
        for _ in 0 ..< rounds {
            _ = full.map { MatryoshkaEmbedding.truncateAndNormalize($0, dimension: 256) }
        }
        var seconds = Date().timeIntervalSince(start)
        let perVector = seconds / Double(rounds * full.count) * 1_000_000
        print(String(format: "[path-bench] Matryoshka 1024→256 每向量 %.1f µs（%d 轮 × %d 条）",
                     perVector, rounds, full.count))

        // 4) 切片本身（语义切片 + 精确 token 计数）
        let article = String(repeating: Self.article, count: 5)
        let options = ChunkingOptions(
            chunkSizeTokens: 800, overlapTokens: 80,
            semanticThreshold: 0.55, enableSemantic: true,
            tokenCounter: counter)
        let chunker = TextChunker(options: options)
        let document = ParsedDocument(text: article, source: "bench.md")
        start = Date()
        let chunks = await chunker.chunk(document: document, variant: .hybrid)
        seconds = Date().timeIntervalSince(start)
        let chunkTokens = chunks.map { counter($0.text) }
        print("[path-bench] 切片 \(article.count) 字 → \(chunks.count) 块，"
            + String(format: "耗时 %.3fs，平均 %.0f token，最大 %d token",
                     seconds,
                     chunkTokens.isEmpty ? 0 : Double(chunkTokens.reduce(0, +)) / Double(chunks.count),
                     chunkTokens.max() ?? 0))

        // 5) 真实切片走生产路径的吞吐
        let realChunks = chunks.map(\.text)
        start = Date()
        let realVectors = try await scheduler.embedDocuments(realChunks)
        seconds = Date().timeIntervalSince(start)
        let realTokens = Double(chunkTokens.reduce(0, +))
        print(String(format: "[path-bench] 真实切片 scheduler+MRL: %.3fs  %.2f chunks/s  %.0f tokens/s",
                     seconds, Double(realChunks.count) / seconds,
                     realTokens / seconds))
        XCTAssertEqual(realVectors.count, realChunks.count)
        XCTAssertEqual(realVectors.first?.count, 256)

        // 6) 真实书籍 markdown 的切片成本（可选）
        if let path = environment["EMBED_PATH_DOC"], !path.isEmpty {
            let book = try String(contentsOfFile: path, encoding: .utf8)
            let bookDocument = ParsedDocument(
                text: book, source: URL(fileURLWithPath: path).lastPathComponent)
            let bookStart = Date()
            let bookChunks = await chunker.chunk(document: bookDocument, variant: .hybrid)
            let bookSeconds = Date().timeIntervalSince(bookStart)
            let bookTokens = bookChunks.map { counter($0.text) }
            let totalTokens = bookTokens.reduce(0, +)
            let sorted = bookTokens.sorted()
            let median = sorted.isEmpty ? 0 : sorted[sorted.count / 2]
            print("[path-bench] 书籍切片 \(book.count) 字 → \(bookChunks.count) 块，"
                + String(format: "耗时 %.2fs，平均 %.0f token，中位 %d，最大 %d，总 token %d",
                         bookSeconds,
                         bookChunks.isEmpty ? 0
                            : Double(totalTokens) / Double(bookChunks.count),
                         median, sorted.last ?? 0, totalTokens))

            // 对照：关闭语义切片，只按 800 token 预算递归切分。
            let recursiveOptions = ChunkingOptions(
                chunkSizeTokens: 800, overlapTokens: 80,
                semanticThreshold: 0.55, enableSemantic: false,
                tokenCounter: counter)
            let recursiveChunker = TextChunker(options: recursiveOptions)
            let recursiveStart = Date()
            let recursiveChunks = await recursiveChunker.chunk(
                document: bookDocument, variant: .hybrid)
            let recursiveSeconds = Date().timeIntervalSince(recursiveStart)
            let recursiveTokens = recursiveChunks.map { counter($0.text) }
            let recursiveSorted = recursiveTokens.sorted()
            print("[path-bench] 对照（语义关闭）→ \(recursiveChunks.count) 块，"
                + String(format: "耗时 %.2fs，平均 %.0f token，中位 %d，最大 %d，总 token %d",
                         recursiveSeconds,
                         recursiveChunks.isEmpty ? 0
                            : Double(recursiveTokens.reduce(0, +)) / Double(recursiveChunks.count),
                         recursiveSorted.isEmpty ? 0 : recursiveSorted[recursiveSorted.count / 2],
                         recursiveSorted.last ?? 0,
                         recursiveTokens.reduce(0, +)))

            // 抽查内置 potion 语义模型是否可用（不可用则语义切片退化为词法相似度）。
            let probe = Array(TextAnalyzer.sentences(String(book.prefix(4_000))).prefix(80))
            let cosines = PotionSentenceSimilarity().adjacentCosines(probe)
            let sample = cosines?.prefix(6).map { String(format: "%.2f", $0) } ?? []
            print("[path-bench] 语义相似度可用: \(cosines != nil)；样本 \(sample)")
            let sentenceCount = TextAnalyzer.sentences(book).count
            print("[path-bench] 全书句子数 \(sentenceCount)，语义切片块数 \(bookChunks.count)"
                + "（比值 \(String(format: "%.2f", Double(bookChunks.count) / Double(max(sentenceCount, 1))))）")

            if !bookChunks.isEmpty {
                print(String(format: "[path-bench] 按实测 %.0f tokens/s 估算 embedding 需 %.1f 分钟",
                             realTokens / max(seconds, 0.001),
                             Double(totalTokens) / max(realTokens / max(seconds, 0.001), 1) / 60))
            }
        }
    }

    private static func makeTexts(counter: (String) -> Int, count: Int,
                                  targetTokens: Int) -> [String] {
        let paragraph = """
        云计算平台通过容器编排实现弹性伸缩，监控体系覆盖指标、日志与链路三个维度，\
        容量评估依据历史峰值与增长趋势建模，并预留合理的安全水位。\
        数据同步需要处理增量更新与冲突合并，检索链路则关注切片质量、向量维度、\
        量化误差与召回排序。工程团队还要权衡吞吐、内存占用与恢复时间，\
        在发布流程中引入灰度、回滚和可观测性，确保长稳运行。
        """
        let base = String(repeating: paragraph, count: 40)
        var characters = min(base.count, targetTokens * 2)
        var sample = String(base.prefix(characters))
        while characters > 40, counter(sample) > targetTokens {
            characters -= 20
            sample = String(base.prefix(characters))
        }
        while characters < base.count, counter(sample) < targetTokens - 10 {
            characters += 10
            sample = String(base.prefix(characters))
        }
        return (0 ..< count).map { index in
            "文档\(index + 1)：\(sample)"
        }
    }

    private static let article = """
    # 多肉植物养护指南

    ## 光照
    多肉植物喜欢充足的散射光。春秋季节可以放在朝南的窗台，每天接受四到六小时日照；
    夏季正午需要遮阴，避免叶片被强光灼伤。室内养护时应定期转动花盆，防止植株向单侧徒长。

    ## 浇水
    浇水遵循干透浇透的原则。春秋生长季大约七到十天浇一次；夏季高温时植株进入半休眠，
    应减少到两到三周一次，并避开正午高温时段。冬季低于五摄氏度时要严格控水，
    保持盆土偏干，防止根系冻伤。

    ## 配土
    配土要保证透气排水。常用配方是颗粒土、泥炭和珍珠岩按六比三比一混合，
    颗粒可以选择赤玉土、火山岩或麦饭石。上盆前在盆底铺一层陶粒作为排水层，
    避免积水导致烂根。

    ## 通风
    通风是多肉安全度夏的关键。封闭阳台在闷热天气要开窗或使用小风扇促进空气流动，
    降低叶面和盆土湿度。发现叶片化水或茎部发黑时，应立即切除病部、消毒并更换干燥介质。

    ## 施肥
    多肉对养分需求不高。生长季每个月施一次稀释的多肉专用液肥即可，
    浓度控制在推荐值的一半；夏季高温和冬季低温停止施肥，避免肥害烧根。
    新上盆的植株缓苗两周后再开始施肥。

    ## 换盆
    根系长满花盆或土壤板结时需要换盆，通常在春秋两季进行。脱盆后轻轻抖掉旧土，
    剪除枯根和烂根，晾根一到两天再上盆。换盆后放在通风阴凉处缓苗，一周内不要浇透水。

    ## 繁殖
    叶插和枝插是最常见的繁殖方式。选择饱满健康的叶片，平放在微潮的颗粒土上，
    避免直射光，两三周后会生根发芽。枝插需要等切口晾干结痂后再插入土中，
    保持基质微潮即可。

    ## 病虫害
    常见害虫有蚧壳虫、红蜘蛛和蚜虫。少量发生时可以用棉签蘸酒精清除，
    严重时使用专用药剂并交替用药，避免产生抗药性。定期检查叶背和叶心，早发现早处理。

    ## 季节管理
    春季适合换盆和扦插，浇水逐步恢复；夏季重点遮阴、通风和控水；
    秋季温差增大，是多肉上色的好时机，可以适当增加光照；冬季搬入室内，
    保持五摄氏度以上并减少浇水。梅雨季节要特别注意防雨和防潮，避免长时间淋雨。

    ## 常见误区
    频繁浇水、长期放在阴暗角落、使用无排水孔的花盆，都会增加烂根风险。
    叶片发软不一定是缺水，也可能是根系受损，应先检查盆土和根部再决定是否补水。
    新买的多肉不要立刻换盆和暴晒，先在通风散射光处缓苗一周更稳妥。
    """
}

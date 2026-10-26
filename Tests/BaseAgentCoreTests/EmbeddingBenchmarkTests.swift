import XCTest
import Foundation
import MLX
@testable import BaseAgentCore

/// bge-small-zh-v1.5-mlx 与 nomic-embed-text-v1.5-mlx-int8 的对比基准。
///
/// 需要真实模型与语料，默认跳过；设置 `EMBED_BENCH=1` 启用。
/// 可选环境变量：
/// - `EMBED_BENCH_MODELS_DIR`：模型根目录（默认应用目录 models）
/// - `EMBED_BENCH_CORPUS`：语料目录（默认 SSD 工作区 rag_data/documents）
/// 报告输出到 `/tmp/embed_bench_report.md` 并打印摘要。
final class EmbeddingBenchmarkTests: XCTestCase {

    struct Target {
        let query: String
        /// 语料中目标块的签名（去空白后包含匹配），用于排名判定。
        let signature: String
        let passage: String
    }

    private static let instruction = "为这个句子生成表示以用于检索相关文章："

    /// 目标块均为真实书籍原文（教育心理学 / 公共基础知识 / 城市发展史 / .NET之美）。
    static let targets: [Target] = [
        Target(
            query: "埃里克森人格发展理论中，儿童面临的第一个基本冲突是什么？",
            signature: "一、埃里克森的社会化发展理论",
            passage: #"一、埃里克森的社会化发展理论 埃里克森把人格发展看作是一个渐进的过程，并将其扩展到人的生命过程，认为人格的发展要经历顺序不变的八个阶段，在每个阶段中个体均面临一种发展危机及发展任务。1.基本的信任感对基本的不信任感（0—1.5岁）该阶段的发展任务是发展对周围世界，尤其是对社会环境的基本态度，培养信任感。儿童面临的第一个基本冲突是信任对不信任。这种信任或不信任的态度在以后的发展阶段中，将由对父母而扩展到对其他人。因此，个体自婴儿期起与父母之间所建立的以信任为基础的社会关系，将是他以后与他人建立良好人际关系的人格基础。2.自主感对羞耻感和怀疑（2—3岁）该阶段的发展任务是培养自主性。本阶段幼儿开始表现出自我控制的需要与倾向、渴望自主的状态，他们对周围的事物感到好奇，喜欢自己动手，能凭借自己的力量做越来越多的事情。这一阶段发展顺利者能按社会要求表现目的性行为；发展障碍者的心理特征是缺乏信心，行动畏首畏尾。3.主动感对内疚感（4—5岁）该阶段的发展任务是培养主动性。由于身体活动能力和语言的发展，儿童有可能把他的活动范围扩展到家庭之外。然而有限的能力常使儿童的身体活动受到成人的限制，使他们感到内疚，从而可能会降低活动的热情。因此，本阶段的危机在于儿童既要发展自己的主动性，又要避免那些可能会被禁止的活动。发展顺利者的心理特征是主动好奇，行动有方向，开始有责任感；发展障碍者的心理特征是畏惧退缩，缺少自我价值感。4.勤奋感对自卑感（6—11岁）该阶段的发展任务是培养勤奋感。这个时期，绝大多数儿童已经进入学校，第一次接受社会赋予他并期望他完成的社会任务。如果儿童在学习中及其他集体活动中获得更多的成功经验，就有利于儿童养成勤奋进取的人格，而其中的挫折和失败则导致了自卑感。如果前三阶段对儿童影响最大的是家庭因素，那么这一阶段对儿童影响最大的就是学校因素。本阶段形成的勤奋进取的性格会延至成年，使其乐于工作并对社会适应良好；若形成自卑心理，往往会成为以后学业颓废的重要原因。5.自我同一性对角色混乱（12—18岁）该阶段的发展任务是培养自我同一性。自我同一性是指个体组织自己的动机、能力、信仰及活动经验而形成的有关自我的一致性形象。在青少年期，"我是谁"成为一个重要问题。他们能够把自己的各种形象综合成一个有意义的整体，对自己的过去、现在、将来产生一种"内在相同和连续"之感。同时，他们也开始认识自己和别人的异同，认识现在和未来在社会中的关联。这一阶段发展顺利者的心理特征是有了明确的自我观念和自我追求的方向；发展障碍者的心理特征是生活无目的、无方向，时常感到彷徨迷茫。6.亲密感对孤独感（成年早期）本阶段的发展任务是建立新型人际关系，克服亲密感和孤独感之间的冲突。这一时期是人们进行求爱和过早期家庭生活的时期。在这一时期，个体往往十分关注镜像自我、注重印象管理、关注自身的前途和发展。发展顺利者与人相处有亲密感；发展障碍者与社会疏离，感到寂寞孤独。7.繁殖感对停滞感（成年中期）本阶段的发展任务是显示自己的生产力、创造力等基本能力，把它用于抚养下一代和关怀社会。发展顺利者的心理特征是热爱家庭、关心社会，有责任感和义务感；发展障碍者的心理特征是不关心别人和社会，缺少生活意义。8.自我整合对绝望感（成年晚期）这一阶段个体的发展受前几个阶段发展的影响极大。人们回顾自己的一生，解决最终的同一性危机。发展顺利者会随心所欲、安享晚年；发展障碍者的心理特征是悔恨旧事，在绝望中度过余生。"#
        ),
        Target(
            query: "教师职业倦怠的特点有哪些？",
            signature: "一、职业倦怠的概念",
            passage: #"一、职业倦怠的概念 职业倦怠，是指在职业环境中，对长期的情绪紧张源和人际关系紧张源的应激反应而表现的一系列心理、生理综合征。二、职业倦怠的特点 1.耗竭感 耗竭感指个体感到自己的能量和资源耗尽、用完的一种心理特征，它主要表现在生理耗竭和情感耗竭两个方面。生理耗竭表现为极度的慢性疲劳、力不从心、疲乏虚弱、睡眠障碍等；情感衰竭特指丧失工作热情、情绪波动大，容易迁怒他人，感到自己的感情处于极度疲劳状态。2.去人格化 去人格化指刻意在自身和工作对象间保持距离，对工作对象和环境采用冷漠和忽视的态度。教师表现一般为：减少接触或拒绝接纳学生；对待有些学生像对待没有生命的物体一样；用带有蔑视色彩的称谓称呼他们；用标签式语言来描述学生。3.低个人成就感 低个人成就感指倾向于消极地评价自己、个人成就感降低、自我效能感下降，对自己工作的意义和价值的评价下降，工作变得机械化且效率低下，缺乏适应性。"#
        ),
        Target(
            query: "教师的班级管理能力主要包括哪些方面？",
            signature: "（2）教师的班级管理能力",
            passage: #"（2）教师的班级管理能力。班级管理能力，是指教师指引和带领班集体实现教育目标的能力。良好的班级管理有利于教学秩序的正常进行，有利于学生的组织管理能力的培养，有利于学生的集体主义与合作精神的形成。班级管理的能力主要包括以下几个方面：选拔与培养学生干部的能力；班级环境的设计能力；班级活动的组织能力。（3）教学能力。教学能力是教师普遍具有的运用特定教材从事教学活动、完成教学任务的能力。教师的教学能力表现在以下三个方面：课前准备能力，课堂教学能力，批改作业和课后辅导能力。"#
        ),
        Target(
            query: "中小学教师职业道德规范中关于为人师表和终身学习的要求是什么？",
            signature: "五、为人师表",
            passage: #"五、为人师表。坚守高尚情操，知荣明耻，严于律己，以身作则。衣着得体，语言规范，举止文明。关心集体，团结协作，尊重同事，尊重家长。作风正派，廉洁奉公。自觉抵制有偿家教，不利用职务之便谋取私利。六、终身学习。崇尚科学精神，树立终身学习理念，拓宽知识视野，更新知识结构。潜心钻研业务，勇于探索创新，不断提高专业素养和教育教学水平。"#
        ),
        Target(
            query: "中法战争后清政府签订了哪些不平等条约？",
            signature: "1883年12月至1885年4月，法国侵略越南",
            passage: #"1883年12月至1885年4月，法国侵略越南并进而侵略中国，史称"中法战争"。中国不败而败，签订了《中法新约》《中法越南边界通商章程》《中法界务条约》《中法续议商务条约》等一系列不平等条约，使法国得到很多权益。受此战的影响，清廷于台湾设省，并积极筹建北洋水师。1894年8月，日本发动了甲午战争；1895年，中日签订了《马关条约》，使我国社会的半殖民地半封建化程度进一步加深。1900年6月，八国联军发动了侵华战争；8月，八国联军入侵北京。1901年，清政府被迫同英、德等11国签订了《辛丑条约》，它标志着我国完全沦为半殖民地半封建社会。"#
        ),
        Target(
            query: "我国第一部社会主义性质的宪法是在哪次会议上通过的？",
            signature: "大会通过的《中华人民共和国宪法》是我国第一部社会主义性质的宪法",
            passage: #"大会通过的《中华人民共和国宪法》是我国第一部社会主义性质的宪法，它明确规定了我国的性质和根本政治制度以及向社会主义过渡的途径、公民的权利和义务、民族区域自治和外交原则，为发扬社会主义民主和建立社会主义法制奠定了初步基础。第一届人大的召开和《宪法》的通过，结束了政协代行人大职能、《共同纲领》暂代《宪法》的局面，加强了人民民主制度。"#
        ),
        Target(
            query: "古代人类从建造房屋和开挖灌渠中学到的经验后来用在了哪里？",
            signature: "沿尼罗河和幼发拉底河两岸分布的先古村民",
            passage: #"证据表明，沿尼罗河和幼发拉底河两岸分布的先古村民们早已掌握了这种技艺。泥和水具有可塑性和随意性，这是孩童皆懂的道理。古人类从建造房屋、水池、灌渠、运河中所学来的经验，后来被广泛应用到大地造形的各种其他部分中去。事实上，人类驯化野生动植物，人类对自身的驯化，以及人类对自然地形的驯化改造，都是些同时并进的客观过程。简要地说，人类改造大地正是后来形成城市的一个重要组成部分，而且是先于城市而进行的。城市与其周围环境的这种密切联系，不幸却被现代人类瓦解着：他们计划以种种受消费者欢迎的人工形式代替复杂的自然地形和生态联系，而置身于险境。"#
        ),
        Target(
            query: "霍华德对城市规划最大的贡献是什么？",
            signature: "霍华德最大的贡献不在于重新塑造城市的物质形式",
            passage: #"这种新式城镇的某些方面已经在伊利诺斯州的里弗赛德等早期独立自足的郊区社区中预示出来；但是，霍华德最大的贡献不在于重新塑造城市的物质形式，而在于发展这种形式下内在的有机概念；因为霍华德不像帕特里克•格迪斯那样是一位生物学家，然而他却把动态平衡和有机平衡这种重要的生物标准引用到城市中来，就是：城市与乡村在范围更大的生物环境中取得平衡，城市内部各种各样功能的平衡，尤其是通过限制城市的面积、人口数目、居住密度等积极控制发展而取得平衡，而一旦这个社区受到过分增长的威胁以致功能失调时，就能另行繁殖（开拓）新社区。如果一个城市要维持它居民的不可缺少的功能，这个城市必须能显示出自己有机的控制和对任何别的有机体的遏止。"#
        ),
        Target(
            query: "复制大文件时为什么不能创建尺寸正好合适的字节数组？",
            signature: "更常见的情况是传递一个更大的文件",
            passage: #"虽然上面的示例可用，但是更常见的情况是传递一个更大的文件，或者事先无法得知文件大小（Length属性抛出异常），因此也就不能创建一个尺寸正好合适的Byte[]数组，此时只能分批读取和写入，每次只读取部分字节，直到文件尾。本书后面涉及流的操作时都会采用这种方式。只要对上面的代码进行简单修改就可以了，这次复制一个大一点的文件，一首5MB的mp3歌曲。"#
        ),
        Target(
            query: "SoapFormatter 相比 BinaryFormatter 有什么优点和缺点？",
            signature: "SoapFormatter类定义在命名空间下",
            passage: #"SoapFormatter类定义在命名空间下，它的使用方式和BinaryFormatter完全相同。使用SOAP的优点就是可以跨平台，因为SOAP是一个开放的协议，所以使用非Windows平台下的其他程序也可以处理它。但是SOAP是用XML来描述的，因此文件尺寸比较大。"#
        ),
    ]

    /// 长块尾部检索：答案位于 1664 字长块的后半段，bge 512 token 截断后不可见。
    private static let longTailQuery = "成年中期的发展任务是什么？"

    func testBenchmark() async throws {
        guard ProcessInfo.processInfo.environment["EMBED_BENCH"] == "1" else {
            throw XCTSkip("未设置 EMBED_BENCH=1，跳过真实模型基准测试")
        }

        let modelsRoot = Self.resolveModelsRoot()
        let corpusDir = ProcessInfo.processInfo.environment["EMBED_BENCH_CORPUS"]
            ?? "/Volumes/SSD/agent-space/rag_data/book-clean-34e533a6/documents"
        guard FileManager.default.fileExists(atPath: corpusDir) else {
            throw XCTSkip("语料目录不存在：\(corpusDir)")
        }

        // 从语料构建干扰块（确定性采样）。
        let corpus = try Self.buildCorpus(corpusDir: corpusDir)
        XCTAssertGreaterThanOrEqual(corpus.count, 60, "干扰块过少")
        print("[bench] 语料干扰块：\(corpus.count)")

        let bgeDir = modelsRoot.appendingPathComponent("bge-small-zh-v1.5-mlx")
        let nomicDir = modelsRoot.appendingPathComponent("nomic-embed-text-v1.5-mlx-int8")
        for dir in [bgeDir, nomicDir] {
            guard FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("model.safetensors").path) else {
                throw XCTSkip("模型缺失：\(dir.path)")
            }
        }

        Self.configureMetallib()

        var report = "# Embedding 对比基准：bge-small-zh-v1.5-mlx vs nomic-embed-text-v1.5-mlx-int8\n\n"

        // ---------- tokenization 统计 ----------
        report += await Self.tokenizerStats(directory: bgeDir, corpus: corpus, label: "bge-small-zh")
        report += await Self.tokenizerStats(directory: nomicDir, corpus: corpus, label: "nomic-int8")

        // ---------- 加载与速度 ----------
        let bge = MLXEmbeddingProvider(directory: bgeDir, modelName: "bge-small-zh-v1.5-mlx")
        let nomic = MLXEmbeddingProvider(directory: nomicDir,
                                         modelName: "nomic-embed-text-v1.5-mlx-int8")
        defer { Task { await bge.unload(); await nomic.unload() } }

        let sample = Array(corpus.prefix(32))
        let bgeLoad = try await Self.measureLoad(provider: bge, texts: sample)
        let nomicLoad = try await Self.measureLoad(provider: nomic, texts: sample)
        report += "\n## 加载与速度\n\n"
        report += "| 指标 | bge-small-zh-v1.5-mlx | nomic-v1.5-mlx-int8 |\n|---|---|---|\n"
        report += "| 加载+首次推理 (ms) | \(bgeLoad) | \(nomicLoad) |\n"
        report += "| 向量维度 | \(bge.dimension) | \(nomic.dimension) |\n"

        // 批量吞吐（32 块 × 3 轮，去除首轮预热）
        let bgeThroughput = try await Self.measureThroughput(provider: bge, texts: sample)
        let nomicThroughput = try await Self.measureThroughput(provider: nomic, texts: sample)
        report += "| 批量吞吐 32块/批 (chunks/s) | \(bgeThroughput) | \(nomicThroughput) |\n"

        // 单条查询延迟（20 次）
        let queries = Self.targets.map(\.query)
        let bgeQuery = try await Self.measureQueryLatency(provider: bge, query: queries[0])
        let nomicQuery = try await Self.measureQueryLatency(provider: nomic, query: queries[0])
        report += "| 单条查询延迟 (ms) | \(bgeQuery) | \(nomicQuery) |\n"

        // ---------- 检索质量 ----------
        let docTexts = corpus + Self.targets.map(\.passage)
        let bgeDocVectors = try await Self.embedInBatches(provider: bge, texts: docTexts)
        let nomicDocVectors = try await Self.embedInBatches(provider: nomic, texts: docTexts)
        XCTAssertEqual(bgeDocVectors.count, docTexts.count)
        XCTAssertEqual(nomicDocVectors.count, docTexts.count)

        report += "\n## 检索质量（10 条中文 query，语料 \(docTexts.count) 块）\n\n"
        report += "| # | Query | bge top1分 | nomic top1分 | bge命中 | nomic命中 |\n|---|---|---|---|---|---|\n"
        var bgeHits1 = 0, bgeHits3 = 0, nomicHits1 = 0, nomicHits3 = 0
        var bgeRR = 0.0, nomicRR = 0.0
        for (index, target) in Self.targets.enumerated() {
            let bgeRank = try await Self.rank(query: target.query, queryProvider: bge,
                                        docVectors: bgeDocVectors, corpus: docTexts,
                                        signature: target.signature)
            let nomicRank = try await Self.rank(query: target.query, queryProvider: nomic,
                                          docVectors: nomicDocVectors, corpus: docTexts,
                                          signature: target.signature)
            if bgeRank.position == 1 { bgeHits1 += 1 }
            if bgeRank.position <= 3 { bgeHits3 += 1 }
            bgeRR += bgeRank.rr
            if nomicRank.position == 1 { nomicHits1 += 1 }
            if nomicRank.position <= 3 { nomicHits3 += 1 }
            nomicRR += nomicRank.rr
            let mark: (Int) -> String = { $0 == 1 ? "✅1" : ($0 <= 3 ? "🟡\($0)" : "❌\($0)") }
            report += "| \(index + 1) | \(target.query) | \(String(format: "%.3f", bgeRank.score)) " +
                "| \(String(format: "%.3f", nomicRank.score)) | \(mark(bgeRank.position)) " +
                "| \(mark(nomicRank.position)) |\n"
        }
        let total = Double(Self.targets.count)
        report += "\n| 指标 | bge | nomic |\n|---|---|---|\n"
        report += "| Recall@1 | \(bgeHits1)/\(Self.targets.count) | \(nomicHits1)/\(Self.targets.count) |\n"
        report += "| Recall@3 | \(bgeHits3)/\(Self.targets.count) | \(nomicHits3)/\(Self.targets.count) |\n"
        report += "| MRR | \(String(format: "%.3f", bgeRR / total)) | \(String(format: "%.3f", nomicRR / total)) |\n"

        // ---------- bge 官方检索指令对比 ----------
        report += "\n## bge query 检索指令（为这个句子生成表示以用于检索相关文章）对比\n\n"
        var instrHits1 = 0, instrRR = 0.0
        for target in Self.targets {
            let rank = try await Self.rank(query: Self.instruction + target.query, queryProvider: bge,
                                     docVectors: bgeDocVectors, corpus: docTexts,
                                     signature: target.signature)
            if rank.position == 1 { instrHits1 += 1 }
            instrRR += rank.rr
        }
        report += "| 方案 | Recall@1 | MRR |\n|---|---|---|\n"
        report += "| bge 无指令 | \(bgeHits1)/\(Self.targets.count) | \(String(format: "%.3f", bgeRR / total)) |\n"
        report += "| bge + 检索指令 | \(instrHits1)/\(Self.targets.count) | \(String(format: "%.3f", instrRR / total)) |\n"
        report += "| nomic（自带前缀） | \(nomicHits1)/\(Self.targets.count) | \(String(format: "%.3f", nomicRR / total)) |\n"

        // ---------- 长块尾部检索（截断影响） ----------
        let longPassage = Self.targets[0].passage
        let bgeLongRank = try await Self.rank(query: Self.longTailQuery, queryProvider: bge,
                                              docVectors: bgeDocVectors, corpus: docTexts,
                                              signature: "7.繁殖感对停滞感")
        let nomicLongRank = try await Self.rank(query: Self.longTailQuery, queryProvider: nomic,
                                                docVectors: nomicDocVectors, corpus: docTexts,
                                                signature: "7.繁殖感对停滞感")
        report += "\n## 长块尾部检索（\(longPassage.count) 字块，答案在后半段）\n\n"
        let bgeLongHit = bgeLongRank.position == Int.max ? "未命中" : "命中"
        let nomicLongHit = nomicLongRank.position == Int.max ? "未命中" : "命中"
        let bgeLongPos = bgeLongRank.position == Int.max ? "-" : "\(bgeLongRank.position)"
        let nomicLongPos = nomicLongRank.position == Int.max ? "-" : "\(nomicLongRank.position)"
        let bgeLongScore = String(format: "%.3f", bgeLongRank.score)
        let nomicLongScore = String(format: "%.3f", nomicLongRank.score)
        report += "| Query：\(Self.longTailQuery) | 命中块 | 排名 | top1 分数 |\n|---|---|---|---|\n"
        report += "| bge（512 token 截断） | \(bgeLongHit) | \(bgeLongPos) | \(bgeLongScore) |\n"
        report += "| nomic（2048 位置） | \(nomicLongHit) | \(nomicLongPos) | \(nomicLongScore) |\n"

        // ---------- 语 义区分度 ----------
        report += "\n## 语义区分度（同义对 vs 无关对 余弦分差）\n\n"
        let paraphraseA = "向量数据库把文本转换成高维向量进行相似度检索"
        let paraphraseB = "Embedding 技术将文本映射为向量以便计算语义相似度"
        let unrelated = "今天天气晴朗，适合出门散步放松心情"
        for (label, provider) in [("bge", bge as EmbeddingProvider), ("nomic", nomic)] {
            let vectors = try await Self.embedInBatches(provider: provider,
                                                        texts: [paraphraseA, paraphraseB, unrelated])
            let same = Self.dot(vectors[0], vectors[1])
            let diff = Self.dot(vectors[0], vectors[2])
            report += "| \(label) | 同义对 \(String(format: "%.3f", same)) | " +
                "无关对 \(String(format: "%.3f", diff)) | 分差 \(String(format: "%.3f", same - diff)) |\n"
        }

        try report.write(toFile: "/tmp/embed_bench_report.md", atomically: true, encoding: .utf8)
        print("[bench] 报告已写入 /tmp/embed_bench_report.md")
        print(report)
    }

    /// 验证 bge-small-zh 已可作为可选方案接入：名称解析 → 工厂构建 → 实际向量化。
    /// 模型目录缺失时跳过。
    func testBGESmallZHWiring() async throws {
        let modelsRoot = Self.resolveModelsRoot()
        let bgeRelative = "models/bge-small-zh-v1.5-mlx"
        guard FileManager.default.fileExists(
            atPath: modelsRoot.appendingPathComponent("bge-small-zh-v1.5-mlx").path) else {
            throw XCTSkip("models/bge-small-zh-v1.5-mlx 不存在，跳过")
        }
        Self.configureMetallib()
        // swift test 的 Bundle.main 不在仓库内，applicationModelsDirectory() 返回 nil；
        // 用注册表自带的覆盖机制锚定模型根目录，等价于生产环境的应用目录 models。
        setenv("BASE_AGENT_MODELS_DIR", modelsRoot.path, 1)
        defer { unsetenv("BASE_AGENT_MODELS_DIR") }

        // 1) 配置值（设置界面写入的 models/<name> 形式）能解析到应用 models 目录。
        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory()
        let directory = try LocalModelRegistry.resolveDirectory(
            kind: .embedding, value: bgeRelative, config: config)
        XCTAssertEqual(directory.lastPathComponent, "bge-small-zh-v1.5-mlx")

        // 2) 工厂按 MLX 后端 + 该路径构建出可用的 provider。
        config.embeddingProvider = "local"
        config.localEmbeddingBackend = "mlx"
        config.localEmbeddingModelPath = bgeRelative
        let provider = LocalModelFactory.shared.makeEmbedding(config: config)
        let mlxProvider = try XCTUnwrap(provider as? MLXEmbeddingProvider,
                                        "应为 MLXEmbeddingProvider：\(provider?.identifier ?? "nil")")
        XCTAssertEqual(mlxProvider.dimension, 512, "bge-small-zh 隐藏维度应为 512")

        // 3) 实际向量化：维度 512、L2 归一化、query 与相关文本相似度高于无关文本。
        let docs = try await mlxProvider.embedDocuments([
            "教师职业倦怠表现为耗竭感、去人格化和低个人成就感。",
            "埃里克森认为人格发展经历八个阶段。",
            "今天天气晴朗，适合出门散步。",
        ])
        XCTAssertEqual(docs.count, 3)
        for vector in docs {
            XCTAssertEqual(vector.count, 512)
            let norm = Self.dot(vector, vector).squareRoot()
            XCTAssertEqual(norm, 1.0, accuracy: 0.02, "L2 归一化")
        }
        let query = try await mlxProvider.embedQuery("教师职业倦怠的特点有哪些？")
        let related = Self.dot(query, docs[0])
        let unrelated = Self.dot(query, docs[2])
        XCTAssertGreaterThan(related, unrelated,
                             "相关文本 \(related) 应比无关文本 \(unrelated) 相似度更高")
        await mlxProvider.unload()
    }

    // MARK: - 工具

    static func resolveModelsRoot() -> URL {
        if let override = ProcessInfo.processInfo.environment["EMBED_BENCH_MODELS_DIR"] {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        // 从本文件路径推导仓库根目录（Tests/BaseAgentCoreTests → 上两级），
        // 不依赖进程 cwd 与 Bundle.main 位置。
        var root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        root = root.deletingLastPathComponent().deletingLastPathComponent()
        if FileManager.default.fileExists(atPath: root.appendingPathComponent("models").path) {
            return root.appendingPathComponent("models", isDirectory: true)
        }
        if let dir = LocalModelRegistry.applicationModelsDirectory() {
            return dir
        }
        return URL(fileURLWithPath: "models", isDirectory: true)
    }

    static func configureMetallib() {
        guard MLX.GPU.metallib == nil else { return }
        var directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<6 {
            let candidate = directory.appendingPathComponent(
                ".build/debug/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib")
            if FileManager.default.fileExists(atPath: candidate.path) {
                MLX.GPU.metallib = candidate
                return
            }
            directory = directory.deletingLastPathComponent()
        }
    }

    /// 按书籍提取 280-650 字块，跨书确定性采样约 90 条干扰块。
    static func buildCorpus(corpusDir: String) throws -> [String] {
        let fm = FileManager.default
        let prefixes = [
            "2015最新版广西中小学教师公开招聘考试专用教材教育心理学",
            "2014最新版陕西省公务员录用考试专用教材公共基础知识",
            "145刘易斯芒福德.城市发展史",
            "_NET之美.NET关键技术深入解析",
        ]
        let signatures = targets.map { normalize($0.signature) }
        var corpus: [String] = []
        for prefix in prefixes {
            let files = (try fm.contentsOfDirectory(atPath: corpusDir))
                .filter { $0.hasPrefix(prefix) && $0.hasSuffix(".md") }
            guard let file = files.first else { continue }
            let text = try String(contentsOfFile: corpusDir + "/" + file, encoding: .utf8)
            let chunks = chunk(text: text)
            let stride = max(1, chunks.count / 23)
            for (index, chunk) in chunks.enumerated() where index % stride == 0 {
                let normalized = normalize(chunk)
                // 排除与目标块重复/近重复的块，保证 ground truth 唯一。
                if signatures.contains(where: { normalized.contains($0) }) { continue }
                corpus.append(chunk)
            }
        }
        return corpus
    }

    private static func chunk(text: String, low: Int = 280, high: Int = 650) -> [String] {
        let paragraphs = text.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        var result: [String] = []
        var buffer = ""
        for paragraph in paragraphs {
            if !buffer.isEmpty, buffer.count + paragraph.count > high {
                result.append(buffer)
                buffer = paragraph
            } else {
                buffer = buffer.isEmpty ? paragraph : buffer + "\n" + paragraph
            }
            if buffer.count >= low, buffer.count <= high {
                result.append(buffer)
                buffer = ""
            }
        }
        if buffer.count >= low { result.append(buffer) }
        return result.filter { $0.count >= 100 }
    }

    static func normalize(_ text: String) -> String {
        text.filter { !$0.isWhitespace }
    }

    private static func tokenizerStats(directory: URL, corpus: [String], label: String) async -> String {
        guard let tokenizer = try? await LocalTokenizer.load(directory: directory) else {
            return "\(label)：tokenizer 加载失败\n"
        }
        let unknownID = tokenizer.base.unknownToken.flatMap { tokenizer.base.convertTokenToId($0) }
        var tokens = 0, unknowns = 0, chars = 0
        for text in corpus.prefix(40) {
            let ids = tokenizer.encode(text, addSpecialTokens: true)
            tokens += ids.count
            chars += text.count
            unknowns += ids.filter { $0 == unknownID }.count
        }
        guard chars > 0 else { return "" }
        return "\(label)：均值 \(String(format: "%.2f", Double(tokens) / Double(chars))) token/字，" +
            "UNK 率 \(String(format: "%.2f%%", Double(unknowns) / Double(tokens) * 100))\n"
    }

    private static func measureLoad(provider: EmbeddingProvider, texts: [String]) async throws -> Int {
        let start = Date()
        _ = try await provider.embedDocuments(texts)
        return Int(Date().timeIntervalSince(start) * 1000)
    }

    private static func measureThroughput(provider: EmbeddingProvider, texts: [String]) async throws -> String {
        _ = try await provider.embedDocuments(texts) // 预热
        let runs = 3
        let start = Date()
        var count = 0
        for _ in 0..<runs {
            _ = try await provider.embedDocuments(texts)
            count += texts.count
        }
        let seconds = Date().timeIntervalSince(start)
        return String(format: "%.1f", Double(count) / seconds)
    }

    private static func measureQueryLatency(provider: EmbeddingProvider, query: String) async throws -> String {
        _ = try await provider.embedQuery(query) // 预热
        let runs = 20
        let start = Date()
        for _ in 0..<runs {
            _ = try await provider.embedQuery(query)
        }
        let ms = Date().timeIntervalSince(start) * 1000 / Double(runs)
        return String(format: "%.1f", ms)
    }

    private static func embedInBatches(
        provider: EmbeddingProvider,
        texts: [String],
        batchSize: Int = 32
    ) async throws -> [[Float]] {
        var vectors: [[Float]] = []
        vectors.reserveCapacity(texts.count)
        var index = 0
        while index < texts.count {
            let end = min(index + batchSize, texts.count)
            vectors.append(contentsOf: try await provider.embedDocuments(Array(texts[index..<end])))
            index = end
        }
        return vectors
    }

    private struct RankResult {
        /// 1-based；未命中为 Int.max。
        let position: Int
        let score: Float
        var rr: Double { position == Int.max ? 0 : 1.0 / Double(position) }
    }

    private static func rank(
        query: String,
        queryProvider: EmbeddingProvider,
        docVectors: [[Float]],
        corpus: [String],
        signature: String
    ) async throws -> RankResult {
        let queryVector = try await queryProvider.embedQuery(query)
        let scored = docVectors.enumerated().map { (index, vector) -> (Int, Float) in
            (index, dot(queryVector, vector))
        }.sorted { $0.1 > $1.1 }
        let needle = normalize(signature)
        let position: Int
        if let rankIndex = scored.firstIndex(where: { normalize(corpus[$0.0]).contains(needle) }) {
            position = rankIndex + 1
        } else {
            position = Int.max
        }
        return RankResult(position: position, score: scored.first?.1 ?? 0)
    }

    private static func dot(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    }
}

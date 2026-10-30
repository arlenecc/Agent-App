import Foundation

// MARK: - 摘要调度模型

/// 摘要任务使用的辅助模型来源。
/// - `resident`：配置已启用且辅助模型可用，直接复用常驻实例，15 分钟/章；
/// - `temporary`：未启用或不可用，空闲会话内临时加载，5 分钟/章。
public enum BookSummaryMode: String, Sendable, Equatable {
    case resident
    case temporary

    public var interval: TimeInterval {
        switch self {
        case .resident: return 900
        case .temporary: return 300
        }
    }
}

/// 空闲会话开始前对候选队列的检查结果。
public enum BookSummaryIdleAvailability: Sendable, Equatable {
    /// 没有待处理文档。
    case noWork
    /// 队首已有可读候选，可以加载辅助模型。
    case ready
    /// 本轮只把不可读候选轮转到队尾，稍后应再次检查。
    case rotated
}

public enum BookSummaryJobStatus: String, Sendable, Equatable {
    /// 仍有章节未摘要。
    case pending
    /// 全部章节已完成，等待生成全书摘要。
    case summarizingBook = "summarizing_book"
    case completed
    /// 本次处理失败，保留 next_section 供下次重试。
    case failed
}

// MARK: - 持久化记录

public struct ChapterSummaryRecord: Sendable, Equatable {
    public var docID: String
    public var sectionIndex: Int
    public var heading: String
    public var title: String
    public var level: Int
    public var summary: String
    public var contentHash: String
    public var updatedAt: Double

    public init(docID: String, sectionIndex: Int, heading: String = "",
                title: String = "", level: Int = 1, summary: String = "",
                contentHash: String = "", updatedAt: Double = Date().timeIntervalSince1970) {
        self.docID = docID
        self.sectionIndex = sectionIndex
        self.heading = heading
        self.title = title
        self.level = level
        self.summary = summary
        self.contentHash = contentHash
        self.updatedAt = updatedAt
    }
}

public struct ChapterSegmentRecord: Sendable, Equatable {
    public var docID: String
    public var sectionIndex: Int
    public var segmentIndex: Int
    public var segmentHash: String
    public var summary: String
    public var updatedAt: Double

    public init(docID: String, sectionIndex: Int, segmentIndex: Int,
                segmentHash: String = "", summary: String = "",
                updatedAt: Double = Date().timeIntervalSince1970) {
        self.docID = docID
        self.sectionIndex = sectionIndex
        self.segmentIndex = segmentIndex
        self.segmentHash = segmentHash
        self.summary = summary
        self.updatedAt = updatedAt
    }
}

public struct BookSummaryRecord: Sendable, Equatable {
    public var docID: String
    public var summary: String
    public var outline: String
    public var chapterCount: Int
    public var sourceHash: String
    public var updatedAt: Double

    public init(docID: String, summary: String = "", outline: String = "",
                chapterCount: Int = 0, sourceHash: String = "",
                updatedAt: Double = Date().timeIntervalSince1970) {
        self.docID = docID
        self.summary = summary
        self.outline = outline
        self.chapterCount = chapterCount
        self.sourceHash = sourceHash
        self.updatedAt = updatedAt
    }
}

public struct BookSummarySegmentRecord: Sendable, Equatable {
    public var docID: String
    public var groupIndex: Int
    public var groupHash: String
    public var summary: String
    public var updatedAt: Double

    public init(docID: String, groupIndex: Int, groupHash: String = "",
                summary: String = "", updatedAt: Double = Date().timeIntervalSince1970) {
        self.docID = docID
        self.groupIndex = groupIndex
        self.groupHash = groupHash
        self.summary = summary
        self.updatedAt = updatedAt
    }
}

public struct SummaryJobRecord: Sendable, Equatable {
    public var docID: String
    public var status: BookSummaryJobStatus
    public var nextSection: Int
    public var totalSections: Int
    public var lastError: String
    public var updatedAt: Double

    public init(docID: String, status: BookSummaryJobStatus = .pending,
                nextSection: Int = 0, totalSections: Int = 0,
                lastError: String = "",
                updatedAt: Double = Date().timeIntervalSince1970) {
        self.docID = docID
        self.status = status
        self.nextSection = nextSection
        self.totalSections = totalSections
        self.lastError = lastError
        self.updatedAt = updatedAt
    }
}

public struct DocumentSummaryProgress: Sendable, Equatable {
    public var docID: String
    public var chapterCount: Int
    public var completedChapters: Int
    public var hasBookSummary: Bool
    public var jobStatus: BookSummaryJobStatus?

    public init(docID: String, chapterCount: Int = 0, completedChapters: Int = 0,
                hasBookSummary: Bool = false, jobStatus: BookSummaryJobStatus? = nil) {
        self.docID = docID
        self.chapterCount = chapterCount
        self.completedChapters = completedChapters
        self.hasBookSummary = hasBookSummary
        self.jobStatus = jobStatus
    }
}

// MARK: - 章节结构

/// 用于摘要的章节：正常情况下 `index` 与 `ResolvedBookOutline.nodes` 一一对应；
/// 无标题纯正文会合成一个不写入目录的「全文」章节。
public struct BookSection: Sendable, Equatable {
    public var index: Int
    public var level: Int
    public var title: String
    public var heading: String
    public var body: String
    public var start: Int
    public var end: Int
    public var contentHash: String

    public init(index: Int, level: Int, title: String, heading: String,
                body: String, start: Int, end: Int, contentHash: String) {
        self.index = index
        self.level = level
        self.title = title
        self.heading = heading
        self.body = body
        self.start = start
        self.end = end
        self.contentHash = contentHash
    }
}

public struct ResolvedBookOutline: Sendable, Equatable {
    public var nodes: [OutlineNode]
    public var sections: [BookSection]
    /// `parser` / `regex` / `none`
    public var source: String
    public var digest: String
    public var chaptersJSON: String
    public var hadRepairs: Bool

    public init(nodes: [OutlineNode], sections: [BookSection], source: String,
                digest: String, chaptersJSON: String, hadRepairs: Bool) {
        self.nodes = nodes
        self.sections = sections
        self.source = source
        self.digest = digest
        self.chaptersJSON = chaptersJSON
        self.hadRepairs = hadRepairs
    }
}

// MARK: - 处理结果

public struct ChapterProcessResult: Sendable, Equatable {
    public var sectionIndex: Int
    public var totalSections: Int
    public var chapterCompleted: Bool
    public var bookCompleted: Bool
    public var titleRepaired: Bool
    public var summary: String

    public init(sectionIndex: Int, totalSections: Int, chapterCompleted: Bool,
                bookCompleted: Bool, titleRepaired: Bool, summary: String) {
        self.sectionIndex = sectionIndex
        self.totalSections = totalSections
        self.chapterCompleted = chapterCompleted
        self.bookCompleted = bookCompleted
        self.titleRepaired = titleRepaired
        self.summary = summary
    }
}

public enum BookSummaryTickOutcome: Sendable, Equatable {
    case throttled(waitSeconds: TimeInterval)
    case noWork
    case processedChapter(docID: String, docName: String, sectionIndex: Int,
                          totalSections: Int, chapterCompleted: Bool)
    case processedBook(docID: String, docName: String)
    case failed(docName: String, message: String)

    public var didWork: Bool {
        switch self {
        case .processedChapter, .processedBook: return true
        default: return false
        }
    }
}

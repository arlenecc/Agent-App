import XCTest
@testable import BaseAgentCore

final class BookSummaryStoreTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-summary-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func makeStore(_ name: String = "index.sqlite") throws -> VectorStore {
        try VectorStore(dbPath: directory.appendingPathComponent(name).path,
                        dimension: 2, enableSQLiteVec: false)
    }

    private func doc(_ name: String, path: String? = nil) -> DocumentRecord {
        let mdPath = path ?? directory.appendingPathComponent("\(name).md").path
        return DocumentRecord(
            docID: VectorStore.hashSource(mdPath),
            docName: name, digest: "# 目录", chapters: "[]",
            mdPath: mdPath, contentHash: "hash-\(name)")
    }

    func testSummaryTablesMigrateAndRoundTrip() throws {
        let store = try makeStore()
        let record = doc("book.md")
        try store.upsertDocument(docName: record.docName, digest: record.digest,
                                 chapters: record.chapters, mdPath: record.mdPath,
                                 contentHash: record.contentHash)
        let now = Date().timeIntervalSince1970

        try store.upsertChapterSummary(ChapterSummaryRecord(
            docID: record.docID, sectionIndex: 3, heading: "第一章 > 1.1",
            title: "1.1 引言", level: 2, summary: String(repeating: "摘", count: 60),
            contentHash: "section-hash", updatedAt: now))
        try store.upsertChapterSegment(ChapterSegmentRecord(
            docID: record.docID, sectionIndex: 3, segmentIndex: 1,
            segmentHash: "seg-hash", summary: "段摘要", updatedAt: now))
        try store.upsertBookSummary(BookSummaryRecord(
            docID: record.docID, summary: "全书摘要", outline: "第一章",
            chapterCount: 4, sourceHash: "book-hash", updatedAt: now))
        try store.upsertBookSummarySegment(BookSummarySegmentRecord(
            docID: record.docID, groupIndex: 0, groupHash: "group-hash",
            summary: "组摘要", updatedAt: now))
        try store.upsertSummaryJob(SummaryJobRecord(
            docID: record.docID, status: .summarizingBook, nextSection: 4,
            totalSections: 4, lastError: "", updatedAt: now))
        try store.setSummaryLastTickAt(123.5)

        XCTAssertEqual(try store.chapterSummaries(docID: record.docID).count, 1)
        XCTAssertEqual(try store.chapterSummary(docID: record.docID, sectionIndex: 3)?
            .summary.count, 60)
        XCTAssertEqual(try store.chapterSegments(docID: record.docID, sectionIndex: 3)
            .first?.segmentHash, "seg-hash")
        XCTAssertEqual(try store.bookSummary(docID: record.docID)?.sourceHash, "book-hash")
        XCTAssertEqual(try store.bookSummarySegments(docID: record.docID)
            .first?.groupHash, "group-hash")
        XCTAssertEqual(try store.summaryJob(docID: record.docID)?.status, .summarizingBook)
        XCTAssertEqual(try store.summaryLastTickAt(), 123.5)
        XCTAssertEqual(try store.document(docID: record.docID)?.contentHash, "hash-book.md")
    }

    func testUpsertDocumentInvalidatesSummariesOnlyWhenContentChanges() throws {
        let store = try makeStore()
        let record = doc("book.md")
        try store.upsertDocument(docName: record.docName, digest: record.digest,
                                 chapters: record.chapters, mdPath: record.mdPath,
                                 contentHash: "v1")
        try store.upsertChapterSummary(ChapterSummaryRecord(
            docID: record.docID, sectionIndex: 0, title: "第一章",
            summary: "旧摘要", contentHash: "old"))

        // 同一正文重复 upsert：摘要保留。
        try store.upsertDocument(docName: record.docName, digest: record.digest,
                                 chapters: record.chapters, mdPath: record.mdPath,
                                 contentHash: "v1")
        XCTAssertEqual(try store.chapterSummaries(docID: record.docID).count, 1)

        // 正文变化：章节/分段/全书/任务状态一并失效。
        try store.upsertBookSummary(BookSummaryRecord(docID: record.docID, summary: "旧全书"))
        try store.upsertSummaryJob(SummaryJobRecord(docID: record.docID, status: .completed))
        try store.upsertDocument(docName: record.docName, digest: record.digest,
                                 chapters: record.chapters, mdPath: record.mdPath,
                                 contentHash: "v2")
        XCTAssertTrue(try store.chapterSummaries(docID: record.docID).isEmpty)
        XCTAssertNil(try store.bookSummary(docID: record.docID))
        XCTAssertNil(try store.summaryJob(docID: record.docID))
    }

    func testSummaryCandidatesSkipCompletedBooks() throws {
        let store = try makeStore()
        let finished = doc("finished.md")
        let pending = doc("pending.md")
        let bookOnly = doc("book-only.md")
        // 没有可识别章节、曾被直接标记完成的文档（旧版无标题纯正文）
        // 仍需进入候选，由摘要引擎合成「全文」章节后补出全书摘要。
        let empty = doc("empty.md")
        for item in [finished, pending, bookOnly, empty] {
            try store.upsertDocument(docName: item.docName, digest: item.digest,
                                     chapters: item.chapters, mdPath: item.mdPath,
                                     contentHash: item.contentHash)
        }
        try store.upsertSummaryJob(SummaryJobRecord(
            docID: finished.docID, status: .completed, nextSection: 2, totalSections: 2))
        try store.upsertBookSummary(BookSummaryRecord(
            docID: finished.docID, summary: "完成"))
        try store.upsertSummaryJob(SummaryJobRecord(
            docID: bookOnly.docID, status: .summarizingBook, nextSection: 3, totalSections: 3))
        try store.upsertSummaryJob(SummaryJobRecord(
            docID: empty.docID, status: .completed, nextSection: 0, totalSections: 0))

        let candidates = try store.summaryCandidates(limit: 10)
        XCTAssertEqual(Set(candidates.map(\.docID)),
                       Set([pending.docID, bookOnly.docID, empty.docID]))
    }

    func testLookupDocumentsPrefersExactPathThenBasenameThenUniqueFuzzy() throws {
        let store = try makeStore()
        let aPath = directory.appendingPathComponent("bookA/report.md").path
        let bPath = directory.appendingPathComponent("bookB/report.md").path
        let uniquePath = directory.appendingPathComponent("bookC/unique.md").path
        let rows = [
            DocumentRecord(docID: VectorStore.hashSource(aPath), docName: "bookA/report.md",
                           digest: "", mdPath: aPath),
            DocumentRecord(docID: VectorStore.hashSource(bPath), docName: "bookB/report.md",
                           digest: "", mdPath: bPath),
            DocumentRecord(docID: VectorStore.hashSource(uniquePath), docName: "bookC/unique.md",
                           digest: "", mdPath: uniquePath),
        ]
        for row in rows {
            try store.upsertDocument(docName: row.docName, digest: "", mdPath: row.mdPath)
        }

        XCTAssertEqual(try store.lookupDocuments(query: aPath).map(\.docName), ["bookA/report.md"])
        XCTAssertEqual(Set(try store.lookupDocuments(query: "report").map(\.docName)),
                       ["bookA/report.md", "bookB/report.md"])
        XCTAssertEqual(try store.lookupDocuments(query: "unique").map(\.docName),
                       ["bookC/unique.md"])
        XCTAssertNil(store.getDocumentDigest(docName: "report.md"),
                     "basename 不唯一时旧 API 不应静默选第一个")
        XCTAssertEqual(store.getDocumentDigest(docName: "bookA/report.md")?.docName,
                       "bookA/report.md")
    }

    func testDeleteDocumentCascadesSummaryRows() throws {
        let store = try makeStore()
        let record = doc("gone.md")
        try store.upsertDocument(docName: record.docName, digest: record.digest,
                                 chapters: record.chapters, mdPath: record.mdPath,
                                 contentHash: record.contentHash)
        try store.upsertChapterSummary(ChapterSummaryRecord(
            docID: record.docID, sectionIndex: 0, title: "第一章", summary: "摘要"))
        try store.upsertBookSummary(BookSummaryRecord(docID: record.docID, summary: "全书"))
        try store.upsertSummaryJob(SummaryJobRecord(docID: record.docID))
        try store.deleteDocumentByMdPath(record.mdPath)

        XCTAssertNil(try store.document(docID: record.docID))
        XCTAssertTrue(try store.chapterSummaries(docID: record.docID).isEmpty)
        XCTAssertNil(try store.bookSummary(docID: record.docID))
        XCTAssertNil(try store.summaryJob(docID: record.docID))
    }

    func testMovingDocumentPathWithoutContentChangeRemovesOrphanSummaries() throws {
        let store = try makeStore()
        let oldPath = directory.appendingPathComponent("old/book.md").path
        let newPath = directory.appendingPathComponent("new/book.md").path
        try store.upsertDocument(docName: "book.md", digest: "# 目录", chapters: "[]",
                                 mdPath: oldPath, contentHash: "v1")
        let oldID = VectorStore.hashSource(oldPath)
        try store.upsertChapterSummary(ChapterSummaryRecord(
            docID: oldID, sectionIndex: 0, title: "第一章", summary: "旧路径摘要"))
        try store.upsertSummaryJob(SummaryJobRecord(
            docID: oldID, status: .completed, nextSection: 1, totalSections: 1))

        // 正文未变、仅缓存路径变化：旧文档行会被删除，摘要不能变成孤儿。
        try store.upsertDocument(docName: "book.md", digest: "# 目录", chapters: "[]",
                                 mdPath: newPath, contentHash: "v1")
        let newID = VectorStore.hashSource(newPath)
        XCTAssertNil(try store.document(docID: oldID))
        XCTAssertTrue(try store.chapterSummaries(docID: oldID).isEmpty,
                      "旧 doc_id 的章节摘要应随文档行一起清理")
        XCTAssertNil(try store.summaryJob(docID: oldID))
        XCTAssertNotNil(try store.document(docID: newID))
        XCTAssertTrue(try store.chapterSummaries(docID: newID).isEmpty)
    }

    func testSummaryProgressIgnoresRowsBeyondJobTotal() throws {
        let store = try makeStore()
        let record = doc("book.md")
        try store.upsertDocument(docName: record.docName, digest: record.digest,
                                 chapters: record.chapters, mdPath: record.mdPath,
                                 contentHash: record.contentHash)
        for index in 0...2 {
            try store.upsertChapterSummary(ChapterSummaryRecord(
                docID: record.docID, sectionIndex: index, title: "第\(index + 1)章",
                summary: "摘要\(index)"))
        }
        try store.upsertSummaryJob(SummaryJobRecord(
            docID: record.docID, status: .summarizingBook,
            nextSection: 2, totalSections: 2))

        let progress = try store.summaryProgress(docID: record.docID)
        XCTAssertEqual(progress.chapterCount, 2)
        XCTAssertEqual(progress.completedChapters, 2,
                       "目录收缩后越界的旧摘要不应计入进度")
    }

    func testDeleteChapterSummariesFromIndexRemovesOnlyTail() throws {
        let store = try makeStore()
        let record = doc("book.md")
        try store.upsertDocument(docName: record.docName, digest: record.digest,
                                 chapters: record.chapters, mdPath: record.mdPath,
                                 contentHash: record.contentHash)
        for index in 0...2 {
            try store.upsertChapterSummary(ChapterSummaryRecord(
                docID: record.docID, sectionIndex: index, title: "第\(index + 1)章",
                summary: "摘要\(index)"))
        }
        try store.upsertChapterSegment(ChapterSegmentRecord(
            docID: record.docID, sectionIndex: 2, segmentIndex: 0,
            segmentHash: "tail-hash", summary: "尾段摘要"))

        try store.deleteChapterSummaries(docID: record.docID, fromSectionIndex: 2)
        XCTAssertEqual(try store.chapterSummaries(docID: record.docID)
            .map(\.sectionIndex).sorted(), [0, 1])
        XCTAssertTrue(try store.chapterSegments(docID: record.docID, sectionIndex: 2).isEmpty)
        XCTAssertEqual(try store.chapterSegments(docID: record.docID, sectionIndex: 1).count, 0)
    }

    func testLegacyDocumentsTableGetsContentHashColumn() throws {
        let path = directory.appendingPathComponent("legacy.sqlite").path
        let legacy = try SQLiteDatabase(path: path)
        try legacy.execScript("""
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            doc_name TEXT NOT NULL,
            digest TEXT NOT NULL DEFAULT '',
            chapters TEXT NOT NULL DEFAULT '',
            md_path TEXT NOT NULL DEFAULT '',
            updated_at REAL NOT NULL
        );
        """)
        try legacy.execute("""
        INSERT INTO documents(doc_id, doc_name, digest, chapters, md_path, updated_at)
        VALUES('old', 'old.md', '', '', '', 1)
        """)
        legacy.close()

        let store = try makeStore("legacy.sqlite")
        XCTAssertEqual(try store.document(docID: "old")?.contentHash, "")
        try store.upsertDocument(docName: "old.md", digest: "", mdPath: "",
                                 contentHash: "new-hash")
        XCTAssertEqual(try store.document(docID: VectorStore.hashSource("old.md"))?
            .contentHash, "new-hash")
    }
}

# Legacy Office (doc/xls/ppt) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 让知识库同步与 Markdown 转换支持 Word 97–2003 `.doc`、Excel 97–2003 `.xls`、PowerPoint 97–2003 `.ppt`，输出结构化 Markdown，且完全内置、无新增运行时依赖。

**Architecture:** 新增 `LegacyOfficeParser` 作为唯一注册入口，按扩展名分派到三个独立提取器；`.doc` 走 AppKit `NSAttributedString` + HTMLMarkdown，`.xls`/`.ppt` 共用 `OLE2CompoundFile`，分别解析 BIFF8 与 PowerPoint 记录树。所有输出走 `finish(profile: .structured)`。

**Tech Stack:** Swift 6 / SwiftPM / XCTest / AppKit / SwiftSoup / Foundation，无新增依赖。

---

## 文件结构

- Create: `Sources/BaseAgentCore/RAG/Parsers/LegacyOfficeLimits.swift` — 共享上限、截断辅助。
- Create: `Sources/BaseAgentCore/RAG/Parsers/OLE2CompoundFile.swift` — OLE2 容器（header/FAT/DIFAT/miniFAT/目录/流）。
- Create: `Sources/BaseAgentCore/RAG/Parsers/LegacyWordExtractor.swift` — `.doc` 提取与标题启发式。
- Create: `Sources/BaseAgentCore/RAG/Parsers/LegacyExcelExtractor.swift` — `.xls` BIFF8 解析、数字格式、公式子集。
- Create: `Sources/BaseAgentCore/RAG/Parsers/LegacyPowerPointExtractor.swift` — `.ppt` 记录树、文本 atom、备注。
- Create: `Sources/BaseAgentCore/RAG/Parsers/LegacyOfficeParser.swift` — 统一入口、错误映射、metadata。
- Create: `Tests/BaseAgentCoreTests/LegacyOfficeTestSupport.swift` — OLE2 / BIFF8 / PPT 最小构造器。
- Create: `Tests/BaseAgentCoreTests/OLE2CompoundFileTests.swift`
- Create: `Tests/BaseAgentCoreTests/LegacyWordExtractorTests.swift`
- Create: `Tests/BaseAgentCoreTests/LegacyExcelExtractorTests.swift`
- Create: `Tests/BaseAgentCoreTests/LegacyPowerPointExtractorTests.swift`
- Create: `Tests/BaseAgentCoreTests/LegacyOfficeIntegrationTests.swift`
- Modify: `Sources/BaseAgentCore/RAG/RAGEngine.swift` — `supportedExtensions` 增加三种扩展名。
- Modify: `Sources/BaseAgentCore/RAG/Parsers/ParserProtocol.swift` — 注册 `LegacyOfficeParser`。
- Modify: `Sources/BaseAgentCore/RAG/Parsers/DocumentCleaner.swift` — `.structured` 档位。
- Modify: `Sources/BaseAgentApp/MarkdownConvertView.swift` — 去掉“仅支持现代格式”提示。
- Modify: `README.md`、`docs/软件设计文档.md` — 支持格式说明。

---

### Task 1: OLE2 容器

**Files:**
- Create: `Sources/BaseAgentCore/RAG/Parsers/LegacyOfficeLimits.swift`
- Create: `Sources/BaseAgentCore/RAG/Parsers/OLE2CompoundFile.swift`
- Test: `Tests/BaseAgentCoreTests/LegacyOfficeTestSupport.swift`
- Test: `Tests/BaseAgentCoreTests/OLE2CompoundFileTests.swift`

**接口约定**

```swift
struct OLE2CompoundFile {
    struct Entry { let name: String; let type: UInt8; let startSector: UInt32; let size: UInt64 }
    init(data: Data) throws
    static func open(url: URL) throws -> OLE2CompoundFile
    var streamNames: [String] { get }
    func containsStream(named name: String) -> Bool
    func streamData(named name: String) throws -> Data
}
```

- [x] 步骤 1：写失败测试：`testRegularStream`、`testMiniStream`、`testStreamNameIsCaseInsensitive`、`testInvalidHeader`、`testTruncatedFatRejected`、`testChainCycleRejected`、`testOversizedStreamRejected`。
- [x] 步骤 2：运行 `swift test --filter OLE2CompoundFileTests`，确认因类型不存在而失败。
- [x] 步骤 3：实现测试用 `OLE2TestBuilder`：512 字节扇区、连续扇区分配、miniFAT + mini stream、目录项、FAT/DIFAT 头部。
- [x] 步骤 4：实现 `OLE2CompoundFile`：签名/扇区大小校验、DIFAT/FAT/miniFAT 链、目录项、大小写不敏感查流、循环与越界检测、512MB 文件与 256MB 单流上限。
- [x] 步骤 5：运行测试并提交（`feat(rag): add OLE2 compound file reader`）。

### Task 2: Word `.doc` 提取

**Files:**
- Create: `Sources/BaseAgentCore/RAG/Parsers/LegacyWordExtractor.swift`
- Test: `Tests/BaseAgentCoreTests/LegacyWordExtractorTests.swift`

**接口约定**

```swift
struct LegacyWordExtractor {
    struct Output { let markdown: String; let outline: [OutlineEntry]; let metadata: [String: String] }
    func extract(url: URL) throws -> Output
}
```

- [x] 步骤 1：测试用 `NSAttributedString` + `.docFormat` 生成 `.doc`：标题 18pt / 正文 12pt / 粗体 / 斜体 / 中文 / 列表；断言 Markdown 含 `# 标题`、`**粗体**`、中文正文；无标题样本不产生 `#`；空文档抛错。
- [x] 步骤 2：运行测试确认失败。
- [x] 步骤 3：实现段落分析：字号众数、字号 > 正文的短段落且无句末标点 → 标题候选，按字号降序映射 `#`/`##`/`###`，表格内段落与列表项排除；`NSParagraphStyle.textLists` 与 `•`/`1.` 前缀识别列表。
- [x] 步骤 4：导出 HTML（UTF-8，64MB 上限），SwiftSoup 中将标题候选 `<p>` 提升为 `<h1>`–`<h3>`（跳过 `<td>/<th>`），复用 `HTMLMarkdown.convert(removeBoilerplate: false)`。
- [x] 步骤 5：metadata `engine=doc`、`legacy_format=word-97`、`paragraphs`、`tables`、`heading_detection=font-size`；AppKit 错误映射为带文件名的 `parse`/`unsupported`。
- [x] 步骤 6：运行测试并提交（`feat(rag): extract legacy Word documents`）。

### Task 3: Excel `.xls` 提取

**Files:**
- Create: `Sources/BaseAgentCore/RAG/Parsers/LegacyExcelExtractor.swift`
- Test: `LegacyOfficeTestSupport.swift`（追加 BIFF8 构造器）
- Test: `Tests/BaseAgentCoreTests/LegacyExcelExtractorTests.swift`

**接口约定**

```swift
struct LegacyExcelExtractor {
    struct Output { let markdown: String; let outline: [OutlineEntry]; let metadata: [String: String] }
    func extract(url: URL) throws -> Output
}
```

- [x] 步骤 1：测试构造器生成带 `Workbook` 流的 OLE2：两个 sheet、BOUNDSHEET、SST + CONTINUE、FORMAT/XF、DATEMODE、LABELSST/LABEL/NUMBER/RK/MULRK/FORMULA + STRING/BOOLERR。
- [x] 步骤 2：测试断言：`## Sheet: 名称`、日期 `yyyy-MM-dd`、百分比 `12%`、`=SUM(A2:A5)`、未解析公式保留缓存值、`Book` 流报 BIFF5 不支持、FILEPASS 报加密不支持、行列截断 metadata。
- [x] 步骤 3：实现全局子流解析（BOUNDSHEET/SST+CONTINUE/FORMAT/XF/DATEMODE/FILEPASS）与逐 sheet 单元格记录。
- [x] 步骤 4：实现数字格式（日期/百分比/去尾零）、公式 token 有界反编译（数字/字符串/布尔/引用/区域/四则/常用函数），不支持时计数 `formula_unparsed`。
- [x] 步骤 5：用 `HTMLMarkdown.escapeTableCell` + `renderMarkdownTable` 输出表格；空 sheet 跳过；上限 100 sheet / 20,000 行 / 256 列。
- [x] 步骤 6：运行测试并提交（`feat(rag): extract legacy Excel workbooks`）。

### Task 4: PowerPoint `.ppt` 提取

**Files:**
- Create: `Sources/BaseAgentCore/RAG/Parsers/LegacyPowerPointExtractor.swift`
- Test: `LegacyOfficeTestSupport.swift`（追加 PPT 记录构造器）
- Test: `Tests/BaseAgentCoreTests/LegacyPowerPointExtractorTests.swift`

**接口约定**

```swift
struct LegacyPowerPointExtractor {
    struct Output { let markdown: String; let outline: [OutlineEntry]; let metadata: [String: String] }
    func extract(url: URL) throws -> Output
}
```

- [x] 步骤 1：测试构造器写入 `PowerPoint Document` 流：两页 Slide（标题 + 正文）、SlideListWithText（顺序）、Notes + NotesAtom（备注）、`TextBytesAtom`。
- [x] 步骤 2：测试断言：`## Slide 1：标题` 顺序正确、正文段落、`**演讲者备注**`、TextBytesAtom 解码、空页仍占序号、`PowerPoint 95`/缺失流报错、记录深度/数量上限。
- [x] 步骤 3：实现有界记录树遍历（深度 64、总量 100 万），收集 Slide 容器（SlideAtom slideId）与 Notes 容器（NotesAtom slideIdRef）。
- [x] 步骤 4：实现 `TextHeaderAtom` 类型配对、`TextCharsAtom` UTF-16LE、`TextBytesAtom` Windows-1252→UTF-8 回退、`\r`/`\v` 段落归一化。
- [x] 步骤 5：按 SlideListWithText 的 `SlidePersistAtom.slideId` 排序；缺失时按文件顺序；备注按 `slideIdRef` 挂载。
- [x] 步骤 6：输出 `## Slide N：标题` + 正文 + 备注；metadata `engine=ppt`、`legacy_format=ppt-97`、`slides`、`notes_slides`、`text_atoms`。
- [x] 步骤 7：运行测试并提交（`feat(rag): extract legacy PowerPoint presentations`）。

### Task 5: 统一入口与引擎集成

**Files:**
- Create: `Sources/BaseAgentCore/RAG/Parsers/LegacyOfficeParser.swift`
- Modify: `Sources/BaseAgentCore/RAG/Parsers/ParserProtocol.swift`
- Modify: `Sources/BaseAgentCore/RAG/RAGEngine.swift:457`
- Modify: `Sources/BaseAgentCore/RAG/Parsers/DocumentCleaner.swift:15`
- Test: `Tests/BaseAgentCoreTests/LegacyOfficeIntegrationTests.swift`

- [x] 步骤 1：先写集成测试：registry 对 `doc/xls/ppt` 选 `LegacyOfficeParser`；`RAGEngine.supportedExtensions` 包含三种；`DocumentCleaner.defaultProfile` 为 `.structured`。
- [x] 步骤 2：实现 `LegacyOfficeParser`（512MB 上限、扩展名分派、空正文报错、错误带文件名、`finish(.structured)`、`progress` 消息）。
- [x] 步骤 3：注册解析器并更新支持列表/清洗档位。
- [x] 步骤 4：运行集成测试并提交（`feat(rag): support legacy Office in engine`）。

### Task 6: UI、文档与回归

**Files:**
- Modify: `Sources/BaseAgentApp/MarkdownConvertView.swift:46`
- Modify: `README.md:55`
- Modify: `docs/软件设计文档.md:361` 与 5.10 Office 章节

- [x] 步骤 1：更新转换提示：`.doc/.docx/.xls/.xlsx/.ppt/.pptx` 全部支持。
- [x] 步骤 2：README 与设计文档删除“旧版格式被拒绝”，补充实现与限制。
- [x] 步骤 3：运行 `swift test --filter "OfficeAndHTMLTests|Legacy|OLE2"`，再跑全量 `swift test --disable-sandbox`。
- [x] 步骤 4：提交（`docs(rag): document legacy Office support`）。

---

## 验收对照（spec）

- 引擎发现/目录同步/转换接受三种扩展名 → Task 5。
- `.doc` 标题/列表/粗斜体/表格 → Task 2。
- `.xls` 多 sheet/日期/百分比/公式 → Task 3。
- `.ppt` 顺序/标题正文/备注 → Task 4。
- 内置实现、无子进程/新依赖 → 全部任务。
- 明确错误、不崩溃 → Task 1/3/4/5。
- 现代 Office 无回归 → Task 6 全量测试。

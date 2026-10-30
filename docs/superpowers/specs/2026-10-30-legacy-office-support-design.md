# 旧版 Office（doc/xls/ppt）解析设计

日期：2026-10-30

状态：设计已确认，待实现

## 背景

当前 `OfficeParser` 只支持 OOXML：`supportedExtensions = ["docx", "xlsx", "pptx"]`，通过 ZIPFoundation 解包。旧版 `.doc/.xls/.ppt` 是 OLE2 复合文档，存在三个缺口：

- `RAGEngine.supportedExtensions` 不包含旧格式，知识库扫描、目录同步和 Markdown 转换不会发现这些文件；
- 没有解析器能读取 OLE2 容器，即使绕过扩展名检查也会失败；
- 转换工具提示与设计文档明确写着「仅支持现代 Office 格式」。

用户需要把旧版 Office 文件纳入引擎发现、转换和知识库入库。解析目标为**结构化 Markdown 保真**：保留标题层级、列表、表格、公式文本、幻灯片标题/正文/备注；不追求 Office 的视觉分页与形状位置。

## 目标

1. 引擎发现、目录同步、单文件/目录 Markdown 转换都接受 `.doc`、`.xls`、`.ppt`。
2. `.doc` 输出标题、列表、表格、粗斜体，尽量保留中文正文。
3. `.xls` 输出多个 sheet 的 Markdown 表格，保留单元格值、日期、百分比、常用公式原文与缓存值。
4. `.ppt` 输出幻灯片顺序、标题/正文层级和演讲者备注。
5. 解析全部内置完成，不依赖 LibreOffice、Python、`textutil` 子进程或新增第三方运行时。
6. 损坏、加密、版本过旧、空正文等失败场景有明确错误，不崩溃、不静默截断。
7. 现代 Office 解析行为保持不变，并补齐旧格式的单元/集成测试。

## 非目标

- 不还原视觉排版：分页、形状位置、字号字体、动画、切换效果。
- 不支持 Word 6.0/95、Excel 5.0/95（BIFF5）、PowerPoint 95；遇到时返回明确的「不支持该旧版本」错误。
- 不支持加密/密码保护文件、VBA/宏、嵌入对象、图表、SmartArt、图片内文字（不做 OCR）。
- 不精确还原 Word 命名样式（Heading 1/2/3）；`.doc` 标题层级用字号/粗体启发式。
- 不覆盖 Excel 的全部公式 token；不支持的公式保留缓存值并计数。
- 不重建 PowerPoint 表格网格；表格文字按文本行输出。
- 不修改现有 `.docx/.xlsx/.pptx` 的解析结果、清洗规则或切片策略。
- 不主动重新入库历史上被忽略的旧格式文件；下一次正常同步扫描时自然发现。

## 支持范围

| 扩展名 | 容器/版本 | 验收目标 | 明确不支持 |
| --- | --- | --- | --- |
| `.doc` | OLE2，Word 97–2003 | 正文、标题、列表、表格、粗斜体 | Word 6.0/95、加密、精确命名样式 |
| `.xls` | OLE2，BIFF8（Excel 97–2003） | sheet、值、日期、百分比、常用公式 | BIFF5/Excel 5.0/95、加密、图表 |
| `.ppt` | OLE2，PowerPoint 97–2003 | 幻灯片顺序、标题/正文、备注 | PowerPoint 95、加密、形状排版 |

扩展名统一转小写匹配。`.xls/.ppt` 必须是 OLE2，内容不符时按「扩展名与实际格式不符」报错；`.doc` 交给 AppKit 系统导入器自动识别，因此 RTF/HTML 伪 `.doc` 扩展名仍可读取。不额外引入格式嗅探库。

## 架构

```
DocumentParserRegistry
   └── LegacyOfficeParser（唯一注册的 DocumentParser）
          ├── LegacyWordExtractor       // .doc，AppKit NSAttributedString
          ├── LegacyExcelExtractor      // .xls，OLE2 + BIFF8
          ├── LegacyPowerPointExtractor // .ppt，OLE2 + PowerPoint 记录
          └── OLE2CompoundFile          // xls/ppt 共享容器读取
```

`OLE2CompoundFile` 负责：OLE2 header、DIFAT/FAT、miniFAT/mini stream、目录项与流名解析；支持 512/4096 字节扇区、大小写不敏感的流名、链循环与越界检测。`.xls/.ppt` 只通过它读取容器，不各自实现 OLE2。

新增文件：

- `Sources/BaseAgentCore/RAG/Parsers/LegacyOfficeParser.swift`
- `Sources/BaseAgentCore/RAG/Parsers/OLE2CompoundFile.swift`
- `Sources/BaseAgentCore/RAG/Parsers/LegacyWordExtractor.swift`
- `Sources/BaseAgentCore/RAG/Parsers/LegacyExcelExtractor.swift`
- `Sources/BaseAgentCore/RAG/Parsers/LegacyPowerPointExtractor.swift`
- `Sources/BaseAgentCore/RAG/Parsers/LegacyOfficeLimits.swift`（共享上限与数字格式常量）

不把旧格式分支塞进现有 1000 多行的 `OfficeParser`，避免现代 Office 逻辑被 OLE2/BIFF/PPT 记录解析污染。

## 数据流

### `.doc`

1. 用 `NSAttributedString(url:options:documentAttributes:)` 在 `autoreleasepool` 中读取 OLE2 Word；AppKit 的系统导入器负责 Word 97–2003 的文本、表格和字体属性。
2. 按段落分析 `NSAttributedString`：
   - 以字符数加权的众数字号作为正文字号；
   - 字号明显大于正文、文本较短、且不以句末标点结尾的段落作为标题候选；
   - 不同标题字号降序映射为 `#` / `##` / `###`，最多 3 级；
   - 列表优先使用 `NSParagraphStyle.textLists`，其次识别 `•`、`·`、`▪`、`-`、`1.` 等前缀；
   - 标题候选不包含表格单元格内的段落。
3. 用 `NSAttributedString.data(from:documentAttributes:[.documentType: .html])` 导出 HTML；HTML 保留表格、粗体、斜体。
4. `SwiftSoup.parse` 后按段落顺序把标题候选对应的 `<p>` 提升为 `<h1>`–`<h3>`；表格内段落不提升。
5. 复用 `HTMLMarkdown.convert(document, removeBoilerplate: false)` 生成 Markdown，再交给 `finish(profile: .structured)`。
6. metadata：`engine=doc`、`legacy_format=word-97`、`paragraphs`、`tables`、`heading_detection=font-size`。

若 AppKit 无法读取（加密、损坏、版本过旧），把底层错误转成带文件名的 `AgentError.parse` / `AgentError.unsupported`。

### `.xls`

1. `OLE2CompoundFile` 读取目录，优先找 `Workbook` 流（BIFF8）；只有 `Book` 流视为 BIFF5，返回不支持。
2. 解析全局子流：
   - `BOUNDSHEET`：sheet 名称与工作表流偏移；
   - `SST` + `CONTINUE`：共享字符串表，处理跨记录字符串、富文本 run 和扩展数据；
   - `FORMAT` / `XF`：数字格式索引，识别日期、百分比；
   - `DATEMODE`：1900/1904 日期系统；
   - `FILEPASS`：加密标记，直接返回不支持。
3. 按 `BOUNDSHEET` 偏移逐 sheet 解析：
   - `LABELSST`、`LABEL`、`NUMBER`、`RK`、`MULRK`、`FORMULA` + 后续 `STRING`、`BOOLERR`、`BLANK`、`MULBLANK`；
   - `FORMULA` 优先取缓存值；对数字、字符串、布尔、单元格引用、区域引用、四则运算和常用函数做有界 token 反编译，生成 `=...` 文本；不支持的 token 只保留缓存值并计数；
   - 日期按 `XF`/`FORMAT` 转 ISO `yyyy-MM-dd`；百分比格式转 `12%`；普通数字去掉无意义尾零；
   - 稀疏单元格补空，保证 Markdown 表格行列对齐。
4. 每个 sheet 输出 `## Sheet: 名称` + Markdown 表格；空 sheet 跳过。
5. metadata：`engine=xls`、`legacy_format=biff8`、`sheets`、`formula_unparsed`、`row_limit_hit`、`column_limit_hit`。

### `.ppt`

1. `OLE2CompoundFile` 找 `PowerPoint Document` 流；缺失则报「不是有效的 PowerPoint 97–2003 文件」。
2. 有界遍历记录树：
   - 记录头 `recVerInstance`（低 4 位版本，高 12 位实例）、`recType`、`recLen`；
   - `recVer == 0xF` 视为容器，递归子记录；
   - 深度上限 64，记录总数上限 100 万，所有偏移做边界检查。
3. 幻灯片顺序优先使用 `SlideListWithText` + `SlidePersistAtom`（含该页文本数量）；结构异常时回退为顺序遍历 `Slide` 容器。
4. 文本 atom：
   - `TextHeaderAtom` 区分标题/正文/备注（标题类型 0/4，正文 1/5/6，备注 2）；
   - `TextCharsAtom` 按 UTF-16LE 解码；`TextBytesAtom` 按 Windows-1252/文档代码页解码，非法字节回退 UTF-8；
   - 同一文本块内的多个 atom 按出现顺序拼接，空文本忽略。
5. 备注来自 notes 文本列表或 `Notes` 容器；按幻灯片顺序挂到对应页。
6. 输出格式与现代 `.pptx` 保持一致：

```markdown
## Slide 1：标题

正文段落

**演讲者备注**

备注内容
```

7. metadata：`engine=ppt`、`legacy_format=ppt-97`、`slides`、`notes_slides`、`text_atoms`。

## Markdown 输出约定

- `.doc`：标题用 `#`/`##`/`###`；列表用 `-` 或 `1.`；表格用标准 Markdown 表格；粗体/斜体用 `**`/`*`。
- `.xls`：sheet 标题为二级标题 `## Sheet: 名称`；单元格内的 `|` 和换行做转义，避免破坏表格。
- `.ppt`：每页一个二级标题 `## Slide N：标题`；正文段落空行分隔；备注用 `**演讲者备注**` 标记。
- 三种格式都通过 `finish(profile: .structured)`，复用现有 `DocumentCleaner` 的 structured 清洗规则。
- 大纲：`.doc` 从标题生成；`.xls` 从 sheet 名生成；`.ppt` 从幻灯片标题生成，缺失标题时用 `Slide N`。

## 引擎集成

需要同步修改：

- `RAGEngine.supportedExtensions` 增加 `doc`、`xls`、`ppt`；
- `DocumentParserRegistry.makeDefault` 注册 `LegacyOfficeParser`；
- `DocumentCleaner.defaultProfile` 把三种旧格式归入 `.structured`；
- `MarkdownConvertView` 去掉「Office 仅支持现代格式」的提示；
- `README.md` 支持格式列表；
- `docs/软件设计文档.md` 的 Office 章节，删除「旧版格式被拒绝」的说明，补充旧格式实现与限制；
- 测试目标补充旧格式用例。

不需要 bump `pipeline_version`：新格式此前未入库，下一次同步扫描会自然发现；现代 Office 解析结果不变。

## 错误处理

| 场景 | 错误类型 | 文案要点 |
| --- | --- | --- |
| 扩展名是 `.xls/.ppt` 但内容不是 OLE2 | `parse` | 扩展名与实际格式不符 |
| OLE2 header/FAT/目录/流损坏或截断 | `parse` | 文件损坏或不是有效 OLE2 |
| Word 导入失败/加密/版本过旧 | `parse` / `unsupported` | 带底层原因与文件名 |
| Excel `FILEPASS` | `unsupported` | 已加密/受密码保护 |
| 只有 `Book` 流（BIFF5） | `unsupported` | 仅支持 Excel 97–2003 |
| PowerPoint 95 记录/加密 | `unsupported` | 仅支持 PowerPoint 97–2003 |
| 解析后正文为空 | `parse` | 未解析出正文 |
| 超过大小/行数/页数上限 | 截断 + metadata | 日志记录截断项，不静默 |

所有错误信息包含文件名；不把底层 `Data` 越界、整数溢出等崩溃暴露给上层。

## 安全与资源上限

| 项 | 上限 |
| --- | --- |
| 单个旧版 Office 文件 | 512 MB |
| OLE2 容器/单流 | 容器 1 GB，单流 256 MB |
| 提取文本总量 | 2000 万字符 |
| `.doc` 导出 HTML | 64 MB |
| `.xls` 每 sheet | 20,000 行 / 256 列（沿用现有 xlsx 上限） |
| `.xls` sheet 数 | 100 |
| `.ppt` 幻灯片数 | 10,000 |
| PPT 记录深度/总数 | 64 / 1,000,000 |
| 公式 token 数量 | 每单元格 4096，整份 100 万 |

实现要求：

- OLE2 sector 偏移、链长度、FAT 索引全部用无符号运算并检查上界，防止回绕；
- 记录遍历检测深度与自引用，避免栈溢出/死循环；
- 大循环定期检查 `Task.isCancelled`；
- 截断时在 metadata 标记 `truncated=true` 与具体原因，日志只记一次，不逐行刷屏。

## 测试

### 单元测试

- `OLE2CompoundFileTests`：正常流、mini stream、大小写流名、损坏 header、截断 FAT、循环链、超出上限。
- `LegacyWordExtractorTests`：标题字号、粗体、斜体、表格、中文、列表、空文档、无标题文档。
- `LegacyExcelExtractorTests`：多 sheet、SST/CONTINUE、数字、日期、百分比、公式缓存值、常用公式原文、行列截断、空 sheet。
- `LegacyPowerPointExtractorTests`：幻灯片顺序、标题/正文、备注、`TextBytesAtom`、空页、记录深度/数量上限。

### 集成测试

- `DocumentParserRegistry` 能为 `doc/xls/ppt` 选中 `LegacyOfficeParser`；
- `RAGEngine.supportedExtensions` 包含三种扩展名；
- `DocumentCleaner.defaultProfile` 返回 `.structured`；
- `MarkdownConversionService` 的单文件与目录校验接受三种扩展名；
- 解析结果能通过 `RAGEngine` 的入库前正文检查（构造 >100 字样本）。

### 测试数据

- `.doc`：用 `NSAttributedString` 的 `.docFormat` 在测试内生成基础样本；表格/列表用提交的小型 `.doc` fixture，fixture 由 `scripts/generate-legacy-office-fixtures` 说明生成方式。
- `.xls`：测试内用最小 OLE2 + BIFF8 构造器生成，不依赖 Python/LibreOffice；覆盖 SST 跨 CONTINUE、日期、百分比、公式。
- `.ppt`：测试内用最小 OLE2 + PowerPoint 记录构造器生成，覆盖两页、标题/正文、备注。
- 所有构造器只用于测试，不进入产品代码。

### 回归

- 现有 `.docx/.xlsx/.pptx` 测试全部通过；
- 全量 `swift test` 通过；
- 手工用真实 Word/Excel/PowerPoint 97–2003 样本各一份，检查 Markdown 可读性与知识库检索。

## 验收标准

1. 在知识库目录放入 `.doc/.xls/.ppt` 后，同步能发现并入库，统计中不再出现「不支持格式」。
2. `.doc` 样本的标题层级、表格、粗斜体、中文正文在 Markdown 中保留。
3. `.xls` 样本的多 sheet、日期、百分比、公式值/常用公式原文正确输出。
4. `.ppt` 样本的幻灯片顺序、标题/正文、备注正确输出。
5. 损坏、加密、BIFF5、PPT95、Word 95 样本返回清晰错误，不崩溃。
6. 现代 Office 解析与全量测试无回归。
7. 不新增运行时依赖，不调用 LibreOffice/Python/textutil 子进程。

## 风险与后续

- `.doc` 标题是字号/粗体启发式，可能把强调短句误判为标题；metadata 记录 `heading_detection=font-size`，后续可增加 Word 二进制 STSH/PAPX 精确样式解析。
- `.xls` 公式 token 只覆盖常用子集；不支持时保留缓存值并计数，后续按真实样本补充 token。
- `.ppt` 依赖 97–2003 记录结构；异常文件回退到顺序遍历，仍失败则报解析错误。
- 测试二进制构造器本身需要维护；若真实样本更稳定，可改为提交小 fixture 并保留构造器用于损坏场景。

import Foundation

// MARK: - 请求与进度

/// 独立文件转 Markdown 的请求。解析/OCR 设置全部来自 `config`，
/// 不读写知识库 manifest，也不触发 embedding/入库。
public struct MarkdownConversionRequest: Sendable {
    public var sourceURL: URL
    public var outputDirectory: URL
    public var config: AgentConfig

    public init(sourceURL: URL, outputDirectory: URL, config: AgentConfig) {
        self.sourceURL = sourceURL
        self.outputDirectory = outputDirectory
        self.config = config
    }
}

public struct MarkdownConversionProgress: Sendable {
    public enum Phase: Sendable, Equatable {
        case scanning
        case parsing
        case writing
        case finished
    }

    public var phase: Phase
    /// 1-based；扫描阶段为 0。
    public var fileIndex: Int
    public var fileTotal: Int
    public var fileName: String
    public var stage: String
    public var pageIndex: Int?
    public var pageTotal: Int?
    public var completedFiles: Int
    public var failedFiles: Int
    /// 0...1 总体进度。
    public var overallFraction: Double

    public init(phase: Phase,
                fileIndex: Int = 0,
                fileTotal: Int = 0,
                fileName: String = "",
                stage: String,
                pageIndex: Int? = nil,
                pageTotal: Int? = nil,
                completedFiles: Int = 0,
                failedFiles: Int = 0,
                overallFraction: Double = 0) {
        self.phase = phase
        self.fileIndex = fileIndex
        self.fileTotal = fileTotal
        self.fileName = fileName
        self.stage = stage
        self.pageIndex = pageIndex
        self.pageTotal = pageTotal
        self.completedFiles = completedFiles
        self.failedFiles = failedFiles
        self.overallFraction = min(1, max(0, overallFraction))
    }
}

public struct MarkdownConversionResult: Sendable, Identifiable {
    public var id: String { source.path }
    public let source: URL
    public let outputURL: URL?
    public let error: String?
    public let elapsed: TimeInterval

    public init(source: URL, outputURL: URL?, error: String?, elapsed: TimeInterval) {
        self.source = source
        self.outputURL = outputURL
        self.error = error
        self.elapsed = elapsed
    }

    public var succeeded: Bool { error == nil && outputURL != nil }
}

public struct MarkdownConversionReport: Sendable {
    public let results: [MarkdownConversionResult]
    public let wasCancelled: Bool
    public let elapsed: TimeInterval

    public init(results: [MarkdownConversionResult],
                wasCancelled: Bool,
                elapsed: TimeInterval) {
        self.results = results
        self.wasCancelled = wasCancelled
        self.elapsed = elapsed
    }

    public var succeededCount: Int { results.filter(\.succeeded).count }
    public var failedCount: Int { results.count - succeededCount }
}

public enum MarkdownConversionError: LocalizedError, Sendable {
    case sourceNotFound(String)
    case sourceNotSupported(String)
    case outputNotDirectory(String)
    case noSupportedFiles(String)
    case cannotPrepareOutput(String)

    public var errorDescription: String? {
        switch self {
        case .sourceNotFound(let path):
            return "待转换路径不存在：\(path)"
        case .sourceNotSupported(let name):
            return "不支持的文件格式：\(name)"
        case .outputNotDirectory(let path):
            return "保存目录不是文件夹：\(path)"
        case .noSupportedFiles(let path):
            return "目录中没有可转换的文件：\(path)"
        case .cannotPrepareOutput(let message):
            return "无法准备保存目录：\(message)"
        }
    }
}

// MARK: - 转换服务

/// 复用 RAG 的 `DocumentParserRegistry`，把单个文件或整个目录转换成
/// 独立的 Markdown 文件。解析、OCR、图片归一化、清洗与入库使用同一套
/// 解析器，唯一区别是不做切片、embedding 和向量库写入。
public actor MarkdownConversionService {
    private struct SourceFile: Sendable {
        let url: URL
        let relativePath: String
    }

    public init() {}

    public func convert(
        request: MarkdownConversionRequest,
        progress: ((MarkdownConversionProgress) -> Void)? = nil,
        onResult: ((MarkdownConversionResult) -> Void)? = nil
    ) async throws -> MarkdownConversionReport {
        let startedAt = Date()
        let fm = FileManager.default
        let source = request.sourceURL.standardizedFileURL
        let output = request.outputDirectory.standardizedFileURL

        var sourceIsDirectory: ObjCBool = false
        guard fm.fileExists(atPath: source.path, isDirectory: &sourceIsDirectory) else {
            throw MarkdownConversionError.sourceNotFound(source.path)
        }

        var outputIsDirectory: ObjCBool = false
        if fm.fileExists(atPath: output.path, isDirectory: &outputIsDirectory) {
            guard outputIsDirectory.boolValue else {
                throw MarkdownConversionError.outputNotDirectory(output.path)
            }
        } else {
            do {
                try fm.createDirectory(at: output, withIntermediateDirectories: true)
            } catch {
                throw MarkdownConversionError.cannotPrepareOutput(error.localizedDescription)
            }
        }

        progress?(MarkdownConversionProgress(
            phase: .scanning,
            stage: "扫描待转换文件",
            overallFraction: 0))

        let files = try Self.discover(source: source,
                                      sourceIsDirectory: sourceIsDirectory.boolValue,
                                      output: output)
        guard !files.isEmpty else {
            throw MarkdownConversionError.noSupportedFiles(source.path)
        }

        let total = files.count
        progress?(MarkdownConversionProgress(
            phase: .scanning,
            fileTotal: total,
            stage: "发现 \(total) 个待转换文件",
            overallFraction: 0))

        let registry = DocumentParserRegistry.makeDefault(config: request.config)
        var results: [MarkdownConversionResult] = []
        var succeeded = 0
        var failed = 0
        var usedOutputs: Set<String> = []
        var cancelled = false

        for (offset, file) in files.enumerated() {
            if Task.isCancelled {
                cancelled = true
                break
            }

            let fileIndex = offset + 1
            let fileName = file.relativePath
            let fileStarted = Date()
            let baseOverall = Double(succeeded + failed) / Double(total)

            progress?(MarkdownConversionProgress(
                phase: .parsing,
                fileIndex: fileIndex,
                fileTotal: total,
                fileName: fileName,
                stage: "解析与提取",
                completedFiles: succeeded,
                failedFiles: failed,
                overallFraction: baseOverall))

            do {
                let options = ParserOptions(config: request.config)
                guard let parser = registry.freshParser(for: file.url,
                                                        options: options) else {
                    throw MarkdownConversionError.sourceNotSupported(fileName)
                }

                let parsed = try await parser.parse(url: file.url, options: options) { event in
                    switch event {
                    case .page(let pageIndex, let pageTotal, let rawStage):
                        // 解析占单文件进度的 90%，留出清洗/写盘时间。
                        let pageFraction = Double(max(0, pageIndex))
                            / Double(max(1, pageTotal))
                        let intra = 0.9 * min(1, max(0, pageFraction))
                        progress?(MarkdownConversionProgress(
                            phase: .parsing,
                            fileIndex: fileIndex,
                            fileTotal: total,
                            fileName: fileName,
                            stage: Self.stageText(rawStage),
                            pageIndex: pageIndex,
                            pageTotal: pageTotal,
                            completedFiles: succeeded,
                            failedFiles: failed,
                            overallFraction: (Double(succeeded + failed) + intra)
                                / Double(total)))
                    case .message(let message):
                        progress?(MarkdownConversionProgress(
                            phase: .parsing,
                            fileIndex: fileIndex,
                            fileTotal: total,
                            fileName: fileName,
                            stage: message,
                            completedFiles: succeeded,
                            failedFiles: failed,
                            overallFraction: baseOverall))
                    }
                }

                if Task.isCancelled {
                    cancelled = true
                    break
                }

                let text = parsed.metadata["cleaned"] == "1"
                    ? parsed.text
                    : DocumentCleaner.clean(
                        parsed.text,
                        keepHeadersFooters: false,
                        profile: DocumentProfile.defaultProfile(for: file.url))
                guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    throw AgentError.parse("解析结果为空")
                }

                progress?(MarkdownConversionProgress(
                    phase: .writing,
                    fileIndex: fileIndex,
                    fileTotal: total,
                    fileName: fileName,
                    stage: "结果清洗 / Markdown 转换",
                    completedFiles: succeeded,
                    failedFiles: failed,
                    overallFraction: (Double(succeeded + failed) + 0.95)
                        / Double(total)))

                let desired = Self.desiredOutputURL(
                    for: file,
                    sourceIsDirectory: sourceIsDirectory.boolValue,
                    output: output)
                let outputURL = Self.uniqueOutputURL(desired, used: &usedOutputs)
                try fm.createDirectory(at: outputURL.deletingLastPathComponent(),
                                       withIntermediateDirectories: true)
                let body = text.hasSuffix("\n") ? text : text + "\n"
                try body.write(to: outputURL, atomically: true, encoding: .utf8)

                succeeded += 1
                let result = MarkdownConversionResult(
                    source: file.url,
                    outputURL: outputURL,
                    error: nil,
                    elapsed: Date().timeIntervalSince(fileStarted))
                results.append(result)
                onResult?(result)

                progress?(MarkdownConversionProgress(
                    phase: .writing,
                    fileIndex: fileIndex,
                    fileTotal: total,
                    fileName: fileName,
                    stage: "已写入 \(outputURL.lastPathComponent)",
                    completedFiles: succeeded,
                    failedFiles: failed,
                    overallFraction: Double(succeeded + failed) / Double(total)))
            } catch {
                if Task.isCancelled || Self.isCancellation(error) {
                    cancelled = true
                    break
                }
                failed += 1
                let message = error.localizedDescription
                let result = MarkdownConversionResult(
                    source: file.url,
                    outputURL: nil,
                    error: message,
                    elapsed: Date().timeIntervalSince(fileStarted))
                results.append(result)
                onResult?(result)
                progress?(MarkdownConversionProgress(
                    phase: .parsing,
                    fileIndex: fileIndex,
                    fileTotal: total,
                    fileName: fileName,
                    stage: "失败：\(message)",
                    completedFiles: succeeded,
                    failedFiles: failed,
                    overallFraction: Double(succeeded + failed) / Double(total)))
            }
        }

        let processed = succeeded + failed
        progress?(MarkdownConversionProgress(
            phase: .finished,
            fileTotal: total,
            stage: cancelled ? "已取消" : "转换完成",
            completedFiles: succeeded,
            failedFiles: failed,
            overallFraction: cancelled ? Double(processed) / Double(total) : 1))

        return MarkdownConversionReport(
            results: results,
            wasCancelled: cancelled,
            elapsed: Date().timeIntervalSince(startedAt))
    }

    // MARK: - 文件发现与输出路径

    private static func discover(source: URL,
                                 sourceIsDirectory: Bool,
                                 output: URL) throws -> [SourceFile] {
        let fm = FileManager.default
        let sourcePath = source.path

        if !sourceIsDirectory {
            let ext = source.pathExtension.lowercased()
            guard RAGEngine.supportedExtensions.contains(ext) else {
                throw MarkdownConversionError.sourceNotSupported(source.lastPathComponent)
            }
            return [SourceFile(url: source, relativePath: source.lastPathComponent)]
        }

        guard let enumerator = fm.enumerator(
            at: source,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]) else {
            throw MarkdownConversionError.sourceNotFound(sourcePath)
        }

        let outputPath = output.path
        let prefix = sourcePath.hasSuffix("/") ? sourcePath : sourcePath + "/"
        let skipOutput = outputPath != sourcePath && outputPath.hasPrefix(prefix)
        var files: [SourceFile] = []

        for case let url as URL in enumerator {
            let values = try? url.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey])
            if values?.isDirectory == true {
                // 输出目录位于待转换目录内时跳过，避免把上次结果再次转换。
                if skipOutput, url.standardizedFileURL.path == outputPath {
                    enumerator.skipDescendants()
                }
                continue
            }
            guard values?.isRegularFile == true else { continue }
            let ext = url.pathExtension.lowercased()
            guard RAGEngine.supportedExtensions.contains(ext) else { continue }

            let path = url.standardizedFileURL.path
            let relative = path.hasPrefix(prefix)
                ? String(path.dropFirst(prefix.count))
                : url.lastPathComponent
            files.append(SourceFile(url: url, relativePath: relative))
        }

        files.sort {
            $0.relativePath.localizedStandardCompare($1.relativePath) == .orderedAscending
        }
        return files
    }

    private static func desiredOutputURL(for file: SourceFile,
                                         sourceIsDirectory: Bool,
                                         output: URL) -> URL {
        let relative = sourceIsDirectory
            ? file.relativePath
            : file.url.lastPathComponent
        // 用 NSString 处理相对路径：URL(fileURLWithPath:) 会把没有目录的
        // 文件名解析到当前工作目录，导致输出路径被污染。
        let nsRelative = relative as NSString
        let directory = nsRelative.deletingLastPathComponent
        // 只对文件名去扩展名，避免把目录一起带进新文件名（sub/sub/b.md）。
        let base = (nsRelative.lastPathComponent as NSString).deletingPathExtension
        let dir = (directory.isEmpty || directory == ".")
            ? output
            : output.appendingPathComponent(directory)
        return dir.appendingPathComponent(base + ".md")
    }

    /// 同一批里 `a.txt` 与 `a.html` 都会得到 `a.md`，后者自动改为 `a-2.md`。
    private static func uniqueOutputURL(_ desired: URL,
                                        used: inout Set<String>) -> URL {
        let standardized = desired.standardizedFileURL
        if used.insert(standardized.path).inserted {
            return desired
        }
        let dir = desired.deletingLastPathComponent()
        let base = desired.deletingPathExtension().lastPathComponent
        var index = 2
        while true {
            let candidate = dir.appendingPathComponent("\(base)-\(index).md")
            if used.insert(candidate.standardizedFileURL.path).inserted {
                return candidate
            }
            index += 1
        }
    }

    private static func stageText(_ stage: String) -> String {
        switch stage {
        case "render": return "页面渲染 / 图片归一化"
        case "text": return "提取文本层"
        case "ocr": return "图片识别"
        case "normalize": return "图片归一化"
        default: return stage.isEmpty ? "解析" : stage
        }
    }

    private static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        if let agentError = error as? AgentError,
           case .cancelled = agentError {
            return true
        }
        return false
    }
}

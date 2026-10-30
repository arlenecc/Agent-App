import Foundation

/// 旧版 Office（Word/Excel/PowerPoint 97–2003）统一解析入口。
///
/// 按扩展名分派到三个内置提取器，统一做大小校验、空正文检查和
/// `.structured` 清洗；不依赖 LibreOffice、Python 或任何子进程。
public final class LegacyOfficeParser: DocumentParser {

    public let name = "LegacyOffice"
    public let supportedExtensions = ["doc", "xls", "ppt"]

    public init() {}

    public func parse(url: URL,
                      options: ParserOptions,
                      progress: ((ParserProgress) -> Void)?) async throws -> ParsedDocument {
        let ext = url.pathExtension.lowercased()
        guard supportedExtensions.contains(ext) else {
            throw AgentError.unsupported("不支持的旧版 Office 格式：\(ext)")
        }
        if Task.isCancelled { throw AgentError.cancelled }

        let values = try? url.resourceValues(forKeys: [.fileSizeKey])
        if let size = values?.fileSize, size > LegacyOfficeLimits.maxFileSize {
            throw AgentError.parse(
                "旧版 Office 文档超过 \(LegacyOfficeLimits.maxFileSize / 1024 / 1024)MB 上限"
                + "（\(url.lastPathComponent)）")
        }

        progress?(.message("解析旧版 \(formatName(ext)) 文档…"))

        let result: (text: String, outline: [OutlineEntry], metadata: [String: String])
        do {
            switch ext {
            case "doc":
                let output = try LegacyWordExtractor().extract(url: url)
                result = (output.markdown, output.outline, output.metadata)
            case "xls":
                let output = try LegacyExcelExtractor().extract(url: url)
                result = (output.markdown, output.outline, output.metadata)
            case "ppt":
                let output = try LegacyPowerPointExtractor().extract(url: url)
                result = (output.markdown, output.outline, output.metadata)
            default:
                throw AgentError.unsupported("不支持的旧版 Office 格式：\(ext)")
            }
        } catch is CancellationError {
            throw AgentError.cancelled
        } catch let error as AgentError {
            throw error
        } catch {
            throw AgentError.parse(
                "旧版 Office 解析失败（\(url.lastPathComponent)）：\(error.localizedDescription)")
        }

        guard !result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AgentError.parse("旧版 Office 文档未解析出正文（\(url.lastPathComponent)）")
        }

        var metadata = result.metadata
        metadata["engine"] = ext
        var parsed = ParsedDocument(text: result.text,
                                    source: url.lastPathComponent,
                                    outline: MarkdownOutline.build(result.outline),
                                    metadata: metadata)
        parsed = finish(parsed, options: options, profile: .structured)
        progress?(.message("旧版 \(formatName(ext)) 文档解析完成"))
        return parsed
    }

    private func formatName(_ ext: String) -> String {
        switch ext {
        case "doc": return "Word"
        case "xls": return "Excel"
        case "ppt": return "PowerPoint"
        default: return "Office"
        }
    }
}

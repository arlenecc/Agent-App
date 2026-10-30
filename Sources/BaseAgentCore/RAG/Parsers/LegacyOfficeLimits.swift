import Foundation

/// 旧版 Office（.doc/.xls/.ppt）解析的共享资源上限。
///
/// 这些格式全部是本地二进制容器，解析时不做流式读取；所有偏移、链长和
/// 文本规模都必须有明确上界，避免损坏文件把内存或 CPU 拖垮。
enum LegacyOfficeLimits {
    /// 单个旧版 Office 文件上限。
    static let maxFileSize = 512 * 1024 * 1024
    /// OLE2 单个流上限（含 mini stream 本体与 WordDocument 等）。
    static let maxStreamSize = 256 * 1024 * 1024
    /// 三种格式最终正文的总字符上限。
    static let maxTextCharacters = 20_000_000
    /// `.doc` 导出中间 HTML 的字节上限。
    static let maxDocHTMLBytes = 64 * 1024 * 1024

    /// `.xls` 每个 sheet 的行/列上限，与现有 `.xlsx` 保持一致。
    static let maxExcelRows = 20_000
    static let maxExcelColumns = 256
    static let maxExcelSheets = 100

    /// `.ppt` 幻灯片数量上限。
    static let maxSlides = 10_000
    /// PowerPoint 记录树深度与总记录数上限。
    static let maxRecordDepth = 64
    static let maxRecordCount = 1_000_000

    /// 公式反编译的 token 上限（单单元格 / 整份工作簿）。
    static let maxFormulaTokensPerCell = 4_096
    static let maxFormulaTokensTotal = 1_000_000

    /// 按字符数截断文本；返回是否发生截断。
    static func limitText(_ text: String, to limit: Int = maxTextCharacters)
        -> (text: String, truncated: Bool) {
        guard limit > 0, text.count > limit else { return (text, false) }
        return (String(text.prefix(limit)), true)
    }

    /// 返回不超过 `limit` 的安全 UTF-16 前缀长度。
    ///
    /// `NSAttributedString` 的长度按 UTF-16 计数，直接截断可能把 emoji 的
    /// 代理对切成半个字符；落点在代理对中间时退回到前一个完整字符。
    static func safeUTF16PrefixLength(_ string: NSString, limit: Int) -> Int {
        guard limit > 0 else { return 0 }
        guard string.length > limit else { return string.length }
        let last = string.character(at: limit - 1)
        if last >= 0xD800 && last <= 0xDBFF {
            return limit - 1
        }
        return limit
    }
}

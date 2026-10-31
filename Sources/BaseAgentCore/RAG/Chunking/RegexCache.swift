import Foundation

/// 线程安全的 `NSRegularExpression` 缓存。
///
/// 标题规则与代码围栏规则在每次 `extract` / `classify` 时都要用同一条 pattern，
/// 重复编译会放大文档结构解析的开销；按 `pattern + options` 复用编译结果。
final class RegexCache: @unchecked Sendable {
    static let shared = RegexCache()

    private struct Key: Hashable {
        let pattern: String
        let options: UInt
    }

    private let lock = NSLock()
    private var cache: [Key: NSRegularExpression] = [:]

    func regex(_ pattern: String,
               options: NSRegularExpression.Options = []) -> NSRegularExpression? {
        let key = Key(pattern: pattern, options: options.rawValue)
        lock.lock()
        if let cached = cache[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()
        guard let regex = try? NSRegularExpression(pattern: pattern, options: options) else {
            return nil
        }
        lock.lock()
        cache[key] = regex
        lock.unlock()
        return regex
    }
}

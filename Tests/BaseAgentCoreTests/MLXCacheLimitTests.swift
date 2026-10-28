import XCTest
import MLX
@testable import BaseAgentCore

/// MLX 空闲 buffer cache 的可配置上限：允许用户试到 2GB，
/// 但仍然是只影响空闲 buffer，不改变权重/KV 的内存硬上限。
final class MLXCacheLimitTests: XCTestCase {
    private let megabyte = 1_024 * 1_024

    func testRuntimeCacheLimitAllowsTwoGB() {
        let original = MLX.Memory.cacheLimit
        defer { MLX.Memory.cacheLimit = original }

        LocalModelFactory.shared.setCacheLimitMB(2_048)
        XCTAssertEqual(MLX.Memory.cacheLimit, 2_048 * megabyte)

        // 超过 2GB 仍要收口，避免配置文件写入离谱数值。
        LocalModelFactory.shared.setCacheLimitMB(4_096)
        XCTAssertEqual(MLX.Memory.cacheLimit, 2_048 * megabyte)

        LocalModelFactory.shared.setCacheLimitMB(128)
        XCTAssertEqual(MLX.Memory.cacheLimit, 256 * megabyte)
    }
}

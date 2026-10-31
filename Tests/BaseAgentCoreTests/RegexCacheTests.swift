import XCTest
@testable import BaseAgentCore

final class RegexCacheTests: XCTestCase {
    func testSamePatternReturnsSameCompiledInstance() {
        let first = RegexCache.shared.regex("^# (.+)$")
        let second = RegexCache.shared.regex("^# (.+)$")
        XCTAssertNotNil(first)
        XCTAssertTrue(first === second, "同一 pattern 必须复用编译结果")
    }

    func testDifferentOptionsUseSeparateEntries() {
        let plain = RegexCache.shared.regex("abc")
        let insensitive = RegexCache.shared.regex("abc", options: [.caseInsensitive])
        XCTAssertNotNil(plain)
        XCTAssertNotNil(insensitive)
        XCTAssertFalse(plain === insensitive)
    }

    func testInvalidPatternReturnsNil() {
        XCTAssertNil(RegexCache.shared.regex("["))
    }
}

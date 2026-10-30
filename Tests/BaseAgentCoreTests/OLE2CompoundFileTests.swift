import XCTest
@testable import BaseAgentCore

final class OLE2CompoundFileTests: XCTestCase {

    func testRegularStream() throws {
        let payload = Data((0..<9000).map { UInt8($0 % 251) })
        let file = try OLE2TestBuilder(streams: [
            .init("Regular", payload),
            .init("Second", Data(repeating: 7, count: 5000)),
        ]).build()
        let compound = try OLE2CompoundFile(data: file.data)

        XCTAssertEqual(try compound.streamData(named: "Regular"), payload)
        XCTAssertEqual(try compound.streamData(named: "Second"),
                       Data(repeating: 7, count: 5000))
        XCTAssertEqual(Set(compound.streamNames), ["Regular", "Second"])
    }

    func testMiniStream() throws {
        let payload = Data((0..<1234).map { UInt8($0 % 97) })
        let file = try OLE2TestBuilder(streams: [
            .init("Mini", payload),
        ]).build()
        let compound = try OLE2CompoundFile(data: file.data)

        XCTAssertEqual(try compound.streamData(named: "Mini"), payload)
        XCTAssertEqual(compound.streamNames, ["Mini"])
    }

    func testMixedMiniAndRegularStreams() throws {
        let mini = Data(repeating: 3, count: 100)
        let regular = Data(repeating: 9, count: 4096)
        let file = try OLE2TestBuilder(streams: [
            .init("Mini", mini),
            .init("Regular", regular),
            .init("Empty", Data()),
        ]).build()
        let compound = try OLE2CompoundFile(data: file.data)

        XCTAssertEqual(try compound.streamData(named: "Mini"), mini)
        XCTAssertEqual(try compound.streamData(named: "Regular"), regular)
        XCTAssertEqual(try compound.streamData(named: "Empty"), Data())
    }

    func testStreamNameIsCaseInsensitive() throws {
        let payload = Data(repeating: 1, count: 6000)
        let file = try OLE2TestBuilder(streams: [
            .init("MixedCase", payload),
        ]).build()
        let compound = try OLE2CompoundFile(data: file.data)

        XCTAssertTrue(compound.containsStream(named: "mixedcase"))
        XCTAssertTrue(compound.containsStream(named: "MIXEDCASE"))
        XCTAssertEqual(try compound.streamData(named: "mixedcase"), payload)
    }

    func testLargeSectorSize() throws {
        let payload = Data(repeating: 5, count: 20_000)
        let file = try OLE2TestBuilder(sectorSize: 4096, streams: [
            .init("LargeSector", payload),
        ]).build()
        let compound = try OLE2CompoundFile(data: file.data)

        XCTAssertEqual(compound.majorVersion, 4)
        XCTAssertEqual(compound.sectorSize, 4096)
        XCTAssertEqual(try compound.streamData(named: "LargeSector"), payload)
    }

    func testInvalidHeader() throws {
        let file = try OLE2TestBuilder(streams: [
            .init("Payload", Data(repeating: 2, count: 5000)),
        ]).build()
        file.setHeaderBytes([0, 0, 0, 0, 0, 0, 0, 0], at: 0)

        XCTAssertThrowsError(try OLE2CompoundFile(data: file.data)) { error in
            XCTAssertTrue(String(describing: error).contains("OLE2"))
        }
    }

    func testInvalidSectorShiftRejected() throws {
        let file = try OLE2TestBuilder(streams: [
            .init("Payload", Data(repeating: 2, count: 5000)),
        ]).build()
        file.setHeaderUInt16(8, at: 0x1E)

        XCTAssertThrowsError(try OLE2CompoundFile(data: file.data))
    }

    func testTruncatedFatRejected() throws {
        let file = try OLE2TestBuilder(streams: [
            .init("Payload", Data(repeating: 2, count: 8192)),
        ]).build()
        // 声明比实际多一个 FAT 扇区；头部 DIFAT 对应表项是 FREESECT。
        file.setHeaderUInt32(UInt32(file.fatSectorIDs.count + 1), at: 0x2C)

        XCTAssertThrowsError(try OLE2CompoundFile(data: file.data))
    }

    func testTruncatedFileRejected() throws {
        let file = try OLE2TestBuilder(streams: [
            .init("Payload", Data(repeating: 2, count: 8192)),
        ]).build()
        file.truncate(to: 512 + file.sectorSize)

        XCTAssertThrowsError(try OLE2CompoundFile(data: file.data))
    }

    func testChainCycleRejected() throws {
        let payload = Data(repeating: 4, count: 8192)
        let file = try OLE2TestBuilder(streams: [
            .init("Cycle", payload),
        ]).build()
        guard let first = file.streamSectors["Cycle"]?.first else {
            return XCTFail("缺少流扇区")
        }
        file.setFATEntry(sector: first, to: first)

        let compound = try OLE2CompoundFile(data: file.data)
        XCTAssertThrowsError(try compound.streamData(named: "Cycle")) { error in
            XCTAssertTrue(String(describing: error).contains("循环"), String(describing: error))
        }
    }

    func testOversizedStreamRejected() throws {
        let file = try OLE2TestBuilder(streams: [
            .init("Huge", Data(repeating: 8, count: 32),
                  forceRegular: true,
                  declaredSize: 300 * 1024 * 1024),
        ]).build()
        let compound = try OLE2CompoundFile(data: file.data)

        XCTAssertThrowsError(try compound.streamData(named: "Huge")) { error in
            XCTAssertTrue(String(describing: error).contains("上限"),
                          String(describing: error))
        }
    }

    func testMissingStreamRejected() throws {
        let file = try OLE2TestBuilder(streams: [
            .init("Present", Data(repeating: 1, count: 20)),
        ]).build()
        let compound = try OLE2CompoundFile(data: file.data)

        XCTAssertFalse(compound.containsStream(named: "Missing"))
        XCTAssertThrowsError(try compound.streamData(named: "Missing"))
    }
}

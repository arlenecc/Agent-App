import Foundation

/// 仅测试使用：构造最小合法 OLE2 复合文档。
///
/// 不依赖任何第三方库，覆盖产品代码需要的结构：FAT/DIFAT 头部、
/// 目录项、普通流、miniFAT 与 mini stream。所有流按连续扇区分配，
/// 便于测试精确修改 FAT 表制造损坏场景。
struct OLE2TestBuilder {

    struct StreamSpec {
        let name: String
        let data: Data
        let forceRegular: Bool
        let declaredSize: UInt64?

        init(_ name: String, _ data: Data,
             forceRegular: Bool = false, declaredSize: UInt64? = nil) {
            self.name = name
            self.data = data
            self.forceRegular = forceRegular
            self.declaredSize = declaredSize
        }
    }

    enum BuildError: Error {
        case invalid(String)
    }

    let sectorSize: Int
    let streams: [StreamSpec]

    init(sectorSize: Int = 512, streams: [StreamSpec]) {
        self.sectorSize = sectorSize
        self.streams = streams
    }

    func build() throws -> OLE2TestFile {
        guard sectorSize == 512 || sectorSize == 4096 else {
            throw BuildError.invalid("扇区大小必须是 512 或 4096")
        }
        let entriesPerSector = sectorSize / 4
        let miniSectorSize = 64
        let cutoff = 4096
        let endOfChain: UInt32 = 0xFFFF_FFFE
        let freeSector: UInt32 = 0xFFFF_FFFF
        let fatSectorMark: UInt32 = 0xFFFF_FFFD

        struct Placement {
            let spec: StreamSpec
            let size: UInt64
            let isMini: Bool
            let isRegular: Bool
            var start: UInt32 = 0xFFFF_FFFE
            var sectors: [UInt32] = []
            var miniSectors: [UInt32] = []
        }

        var placements: [Placement] = []
        var miniFat: [UInt32] = []
        var miniStreamData = Data()

        for spec in streams {
            let declared = spec.declaredSize ?? UInt64(spec.data.count)
            if spec.data.isEmpty {
                placements.append(Placement(spec: spec, size: declared,
                                            isMini: false, isRegular: false))
                continue
            }
            if spec.forceRegular || spec.data.count >= cutoff {
                placements.append(Placement(spec: spec, size: declared,
                                            isMini: false, isRegular: true))
                continue
            }

            let start = UInt32(miniFat.count)
            let count = (spec.data.count + miniSectorSize - 1) / miniSectorSize
            var ids: [UInt32] = []
            for index in 0..<count {
                let current = UInt32(miniFat.count)
                miniFat.append(index + 1 < count ? current + 1 : endOfChain)
                ids.append(current)

                let begin = index * miniSectorSize
                let end = min(begin + miniSectorSize, spec.data.count)
                miniStreamData.append(Data(spec.data[begin..<end]))
                let pad = miniSectorSize - (end - begin)
                if pad > 0 { miniStreamData.append(Data(count: pad)) }
            }
            placements.append(Placement(spec: spec, size: declared,
                                        isMini: true, isRegular: false,
                                        start: start, miniSectors: ids))
        }

        let directoryEntries = 1 + streams.count
        let directoryBytes = ((directoryEntries * 128 + sectorSize - 1) / sectorSize) * sectorSize
        let directorySectorCount = directoryBytes / sectorSize
        let miniFatBytes = ((miniFat.count * 4 + sectorSize - 1) / sectorSize) * sectorSize
        let miniFatSectorCount = miniFatBytes / sectorSize
        let miniStreamSectorCount = (miniStreamData.count + sectorSize - 1) / sectorSize
        let regularSectorCount = placements.reduce(0) { partial, placement in
            guard placement.isRegular else { return partial }
            return partial + (placement.spec.data.count + sectorSize - 1) / sectorSize
        }
        let nonFatSectors = directorySectorCount + miniFatSectorCount
            + miniStreamSectorCount + regularSectorCount

        var fatSectorCount = 1
        while true {
            let total = nonFatSectors + fatSectorCount
            let needed = max(1, (total + entriesPerSector - 1) / entriesPerSector)
            if needed == fatSectorCount { break }
            fatSectorCount = needed
        }
        let totalSectors = nonFatSectors + fatSectorCount
        guard totalSectors <= 0xFFFF_FFF0 else {
            throw BuildError.invalid("测试文件扇区过多")
        }
        guard fatSectorCount <= 109 else {
            throw BuildError.invalid("测试构造器只支持头部 DIFAT")
        }

        var nextSector = UInt32(fatSectorCount)
        let directoryStart = nextSector
        nextSector += UInt32(directorySectorCount)
        let directorySectors = Array(directoryStart..<nextSector)
        let miniFatStart = nextSector
        if miniFatSectorCount > 0 { nextSector += UInt32(miniFatSectorCount) }
        let miniFatSectors = miniFatSectorCount > 0
            ? Array(miniFatStart..<nextSector) : []
        let miniStreamStart = nextSector
        if miniStreamSectorCount > 0 { nextSector += UInt32(miniStreamSectorCount) }
        let miniStreamSectors = miniStreamSectorCount > 0
            ? Array(miniStreamStart..<nextSector) : []

        var regularStreamSectors: [String: [UInt32]] = [:]
        for index in placements.indices where placements[index].isRegular {
            let count = (placements[index].spec.data.count + sectorSize - 1) / sectorSize
            placements[index].start = nextSector
            placements[index].sectors = Array(nextSector..<(nextSector + UInt32(count)))
            regularStreamSectors[placements[index].spec.name] = placements[index].sectors
            nextSector += UInt32(count)
        }
        guard Int(nextSector) == totalSectors else {
            throw BuildError.invalid("测试文件扇区布局不一致")
        }

        var fat = [UInt32](repeating: freeSector,
                           count: fatSectorCount * entriesPerSector)
        for index in 0..<fatSectorCount { fat[index] = fatSectorMark }

        func link(_ sectors: [UInt32]) {
            for (index, sector) in sectors.enumerated() {
                fat[Int(sector)] = index + 1 < sectors.count
                    ? sectors[index + 1] : endOfChain
            }
        }
        link(directorySectors)
        link(miniFatSectors)
        link(miniStreamSectors)
        for sectors in regularStreamSectors.values { link(sectors) }

        var file = Data(count: 512 + totalSectors * sectorSize)
        let major: UInt16 = sectorSize == 4096 ? 4 : 3
        let sectorShift: UInt16 = sectorSize == 4096 ? 12 : 9

        let signature: [UInt8] = [0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1]
        file.replaceSubrange(0..<8, with: signature)
        testWriteUInt16(0x003E, to: &file, at: 0x18)
        testWriteUInt16(major, to: &file, at: 0x1A)
        testWriteUInt16(0xFFFE, to: &file, at: 0x1C)
        testWriteUInt16(sectorShift, to: &file, at: 0x1E)
        testWriteUInt16(6, to: &file, at: 0x20)
        testWriteUInt32(major == 4 ? UInt32(directorySectorCount) : 0,
                        to: &file, at: 0x28)
        testWriteUInt32(UInt32(fatSectorCount), to: &file, at: 0x2C)
        testWriteUInt32(directoryStart, to: &file, at: 0x30)
        testWriteUInt32(0, to: &file, at: 0x34)
        testWriteUInt32(4096, to: &file, at: 0x38)
        testWriteUInt32(miniFatSectorCount > 0 ? miniFatStart : endOfChain,
                        to: &file, at: 0x3C)
        testWriteUInt32(UInt32(miniFatSectorCount), to: &file, at: 0x40)
        testWriteUInt32(endOfChain, to: &file, at: 0x44)
        testWriteUInt32(0, to: &file, at: 0x48)
        for index in 0..<109 {
            let value = index < fatSectorCount ? UInt32(index) : freeSector
            testWriteUInt32(value, to: &file, at: 0x4C + index * 4)
        }

        func sectorOffset(_ sector: UInt32) -> Int {
            512 + Int(sector) * sectorSize
        }

        for index in 0..<fatSectorCount {
            let base = sectorOffset(UInt32(index))
            for entry in 0..<entriesPerSector {
                testWriteUInt32(fat[index * entriesPerSector + entry],
                                to: &file, at: base + entry * 4)
            }
        }

        var directoryItems: [OLE2TestDirectoryEntry] = [
            OLE2TestDirectoryEntry(name: "Root Entry", type: 5,
                                   start: miniStreamSectorCount > 0 ? miniStreamStart : endOfChain,
                                   size: UInt64(miniStreamData.count)),
        ]
        for placement in placements {
            directoryItems.append(OLE2TestDirectoryEntry(name: placement.spec.name,
                                                         type: 2,
                                                         start: placement.start,
                                                         size: placement.size))
        }
        for (index, item) in directoryItems.enumerated() {
            writeDirectoryEntry(item, to: &file,
                                at: sectorOffset(directoryStart) + index * 128)
        }

        let miniFatBase = sectorOffset(miniFatStart)
        for (index, value) in miniFat.enumerated() {
            testWriteUInt32(value, to: &file, at: miniFatBase + index * 4)
        }

        for (index, sector) in miniStreamSectors.enumerated() {
            let begin = index * sectorSize
            let end = min(begin + sectorSize, miniStreamData.count)
            if begin < end {
                file.replaceSubrange(sectorOffset(sector)..<(sectorOffset(sector) + end - begin),
                                     with: miniStreamData[begin..<end])
            }
        }

        for placement in placements where placement.isRegular {
            for (index, sector) in placement.sectors.enumerated() {
                let begin = index * sectorSize
                let end = min(begin + sectorSize, placement.spec.data.count)
                if begin < end {
                    file.replaceSubrange(sectorOffset(sector)..<(sectorOffset(sector) + end - begin),
                                         with: placement.spec.data[begin..<end])
                }
            }
        }

        var starts: [String: UInt32] = [:]
        for placement in placements { starts[placement.spec.name] = placement.start }

        return OLE2TestFile(data: file,
                            sectorSize: sectorSize,
                            fatSectorIDs: (0..<fatSectorCount).map { UInt32($0) },
                            directorySectors: directorySectors,
                            miniFatSectors: miniFatSectors,
                            miniStreamSectors: miniStreamSectors,
                            streamStarts: starts,
                            streamSectors: regularStreamSectors)
    }

    private func writeDirectoryEntry(_ entry: OLE2TestDirectoryEntry,
                                     to file: inout Data, at offset: Int) {
        var nameBytes: [UInt8] = []
        for unit in entry.name.utf16 {
            nameBytes.append(UInt8(truncatingIfNeeded: unit))
            nameBytes.append(UInt8(truncatingIfNeeded: unit >> 8))
        }
        let clipped = Array(nameBytes.prefix(62))
        if !clipped.isEmpty {
            file.replaceSubrange(offset..<(offset + clipped.count), with: clipped)
        }
        testWriteUInt16(UInt16(min(64, clipped.count + 2)), to: &file, at: offset + 0x40)
        file[offset + 0x42] = entry.type
        file[offset + 0x43] = 1
        testWriteUInt32(0xFFFF_FFFF, to: &file, at: offset + 0x44)
        testWriteUInt32(0xFFFF_FFFF, to: &file, at: offset + 0x48)
        testWriteUInt32(0xFFFF_FFFF, to: &file, at: offset + 0x4C)
        testWriteUInt32(entry.start, to: &file, at: offset + 0x74)
        testWriteUInt64(entry.size, to: &file, at: offset + 0x78)
    }
}

/// 构造完成后的测试文件，提供直接改坏关键结构的方法。
final class OLE2TestFile {
    private(set) var data: Data
    let sectorSize: Int
    let fatSectorIDs: [UInt32]
    let directorySectors: [UInt32]
    let miniFatSectors: [UInt32]
    let miniStreamSectors: [UInt32]
    let streamStarts: [String: UInt32]
    let streamSectors: [String: [UInt32]]

    init(data: Data, sectorSize: Int, fatSectorIDs: [UInt32],
         directorySectors: [UInt32], miniFatSectors: [UInt32],
         miniStreamSectors: [UInt32],
         streamStarts: [String: UInt32],
         streamSectors: [String: [UInt32]]) {
        self.data = data
        self.sectorSize = sectorSize
        self.fatSectorIDs = fatSectorIDs
        self.directorySectors = directorySectors
        self.miniFatSectors = miniFatSectors
        self.miniStreamSectors = miniStreamSectors
        self.streamStarts = streamStarts
        self.streamSectors = streamSectors
    }

    func sectorOffset(_ sector: UInt32) -> Int {
        512 + Int(sector) * sectorSize
    }

    func setHeaderBytes(_ bytes: [UInt8], at offset: Int) {
        data.replaceSubrange(offset..<(offset + bytes.count), with: bytes)
    }

    func setHeaderUInt16(_ value: UInt16, at offset: Int) {
        testWriteUInt16(value, to: &data, at: offset)
    }

    func setHeaderUInt32(_ value: UInt32, at offset: Int) {
        testWriteUInt32(value, to: &data, at: offset)
    }

    func setFATEntry(sector: UInt32, to value: UInt32) {
        let perSector = sectorSize / 4
        let fatSector = fatSectorIDs[Int(sector) / perSector]
        let offset = sectorOffset(fatSector) + (Int(sector) % perSector) * 4
        testWriteUInt32(value, to: &data, at: offset)
    }

    func truncate(to byteCount: Int) {
        guard byteCount >= 0, byteCount <= data.count else { return }
        data = Data(data.prefix(byteCount))
    }
}

private struct OLE2TestDirectoryEntry {
    let name: String
    let type: UInt8
    let start: UInt32
    let size: UInt64
}

// MARK: - little endian 写入

private func testWriteUInt16(_ value: UInt16, to data: inout Data, at offset: Int) {
    let bytes: [UInt8] = [UInt8(truncatingIfNeeded: value),
                          UInt8(truncatingIfNeeded: value >> 8)]
    data.replaceSubrange(offset..<(offset + 2), with: bytes)
}

private func testWriteUInt32(_ value: UInt32, to data: inout Data, at offset: Int) {
    let bytes: [UInt8] = [UInt8(truncatingIfNeeded: value),
                          UInt8(truncatingIfNeeded: value >> 8),
                          UInt8(truncatingIfNeeded: value >> 16),
                          UInt8(truncatingIfNeeded: value >> 24)]
    data.replaceSubrange(offset..<(offset + 4), with: bytes)
}

private func testWriteUInt64(_ value: UInt64, to data: inout Data, at offset: Int) {
    let bytes: [UInt8] = [
        UInt8(truncatingIfNeeded: value),
        UInt8(truncatingIfNeeded: value >> 8),
        UInt8(truncatingIfNeeded: value >> 16),
        UInt8(truncatingIfNeeded: value >> 24),
        UInt8(truncatingIfNeeded: value >> 32),
        UInt8(truncatingIfNeeded: value >> 40),
        UInt8(truncatingIfNeeded: value >> 48),
        UInt8(truncatingIfNeeded: value >> 56),
    ]
    data.replaceSubrange(offset..<(offset + 8), with: bytes)
}

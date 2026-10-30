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

// MARK: - BIFF8 工作簿

/// 仅测试使用：构造带 BIFF8 记录的最小 `.xls` 工作簿。
///
/// 覆盖产品解析器需要的全局子流（BOUNDSHEET/SST+CONTINUE/FORMAT/XF/
/// DATEMODE/FILEPASS）与常见单元格记录。所有记录按 BIFF8 小端格式写入，
/// 表格偏移按最终流长度计算。
struct BIFF8TestBuilder {

    struct Sheet {
        let name: String
        let cells: [Cell]
    }

    struct Cell {
        enum Value {
            case labelSST(Int)
            case label(String)
            case number(Double, xf: UInt16)
            case rk(UInt32, xf: UInt16)
            case mulrk(values: [UInt32], xf: UInt16)
            case formula(tokens: [UInt8], result: Data, string: String?, xf: UInt16)
            case boolean(Bool, xf: UInt16)
            case error(UInt8, xf: UInt16)
            case blank(xf: UInt16)
        }

        let row: Int
        let column: Int
        let value: Value
    }

    var sheets: [Sheet]
    var sharedStrings: [String]
    var formats: [(id: UInt16, value: String)]
    var xfFormatIDs: [UInt16]
    var date1904: Bool
    var filePass: Bool
    var streamName: String
    /// 在指定字符串数量后拆出 CONTINUE（不额外写字符编码标志字节）。
    var sstContinueAfterString: Int?
    /// 在共享字符串内部指定字符数处拆出 CONTINUE，并在续段首写编码标志字节。
    var sstSplitInsideString: (index: Int, characters: Int)?

    init(sheets: [Sheet],
         sharedStrings: [String] = [],
         formats: [(id: UInt16, value: String)] = [],
         xfFormatIDs: [UInt16] = [0],
         date1904: Bool = false,
         filePass: Bool = false,
         streamName: String = "Workbook",
         sstContinueAfterString: Int? = nil,
         sstSplitInsideString: (index: Int, characters: Int)? = nil) {
        self.sheets = sheets
        self.sharedStrings = sharedStrings
        self.formats = formats
        self.xfFormatIDs = xfFormatIDs
        self.date1904 = date1904
        self.filePass = filePass
        self.streamName = streamName
        self.sstContinueAfterString = sstContinueAfterString
        self.sstSplitInsideString = sstSplitInsideString
    }

    func build() throws -> OLE2TestFile {
        let sheetPayloads = sheets.map(sheetData)
        let placeholder = globalData(sheetOffsets: Array(repeating: 0, count: sheets.count))

        var offsets: [UInt32] = []
        var next = UInt32(placeholder.count)
        for payload in sheetPayloads {
            offsets.append(next)
            next += UInt32(payload.count)
        }

        var stream = globalData(sheetOffsets: offsets)
        for payload in sheetPayloads { stream.append(payload) }

        return try OLE2TestBuilder(streams: [.init(streamName, stream)]).build()
    }

    // MARK: 全局子流

    private func globalData(sheetOffsets: [UInt32]) -> Data {
        var data = Data()
        data.append(record(type: 0x0809, payload: bofPayload(type: 0x0005)))
        if filePass {
            data.append(record(type: 0x002F, payload: Data([0x00, 0x00, 0x00, 0x00])))
        }
        data.append(record(type: 0x0022,
                           payload: Data(biffBytes(date1904 ? UInt16(1) : UInt16(0)))))
        for format in formats {
            var payload = Data(biffBytes(format.id))
            payload.append(xlUnicodeString(format.value))
            data.append(record(type: 0x041E, payload: payload))
        }
        for formatID in xfFormatIDs {
            data.append(record(type: 0x00E0, payload: xfPayload(formatID: formatID)))
        }
        for (index, sheet) in sheets.enumerated() {
            let offset = index < sheetOffsets.count ? sheetOffsets[index] : 0
            data.append(record(type: 0x0085,
                               payload: boundsheetPayload(name: sheet.name, offset: offset)))
        }
        for sstRecord in sstRecords() { data.append(sstRecord) }
        data.append(record(type: 0x000A, payload: Data()))
        return data
    }

    private func bofPayload(type: UInt16) -> Data {
        var bytes: [UInt8] = []
        bytes += biffBytes(UInt16(0x0600))
        bytes += biffBytes(type)
        bytes += biffBytes(UInt16(0x0DBB))
        bytes += biffBytes(UInt16(0x07CC))
        bytes += biffBytes(UInt32(0x00000041))
        bytes += biffBytes(UInt16(0x0600))
        bytes += [0x00, 0x00]
        return Data(bytes)
    }

    private func xfPayload(formatID: UInt16) -> Data {
        var payload = [UInt8](repeating: 0, count: 20)
        let format = biffBytes(formatID)
        payload[2] = format[0]
        payload[3] = format[1]
        return Data(payload)
    }

    private func boundsheetPayload(name: String, offset: UInt32) -> Data {
        var bytes = biffBytes(offset)
        bytes += [0x00, 0x00]
        let units = Array(name.utf16)
        bytes += [UInt8(min(units.count, 255)), 0x01]
        for unit in units.prefix(255) { bytes += biffBytes(unit) }
        return Data(bytes)
    }

    // MARK: SST / CONTINUE

    private struct SSTParts {
        let units: [UInt16]
        let flags: UInt8
        let compressed: Bool
    }

    private func sstRecords() -> [Data] {
        guard !sharedStrings.isEmpty else { return [] }
        var header = Data(biffBytes(UInt32(sharedStrings.count)))
        header.append(contentsOf: biffBytes(UInt32(sharedStrings.count)))

        let parts = sharedStrings.map(sstParts)
        if let split = sstSplitInsideString,
           split.index >= 0, split.index < parts.count {
            let target = parts[split.index]
            let splitCount = min(max(0, split.characters), target.units.count)
            if splitCount > 0, splitCount < target.units.count {
                var first = header
                for index in 0..<split.index { first.append(sstPayload(parts[index])) }
                first.append(sstHeader(target))
                first.append(sstChars(target, start: 0, end: splitCount))

                var continuation = Data([target.flags])
                continuation.append(sstChars(target, start: splitCount,
                                             end: target.units.count))
                for index in (split.index + 1)..<parts.count {
                    continuation.append(sstPayload(parts[index]))
                }
                return [record(type: 0x00FC, payload: first),
                        record(type: 0x003C, payload: continuation)]
            }
        }

        if let split = sstContinueAfterString,
           split > 0, split < parts.count {
            var first = header
            for index in 0..<split { first.append(sstPayload(parts[index])) }
            var continuation = Data()
            for index in split..<parts.count {
                continuation.append(sstPayload(parts[index]))
            }
            return [record(type: 0x00FC, payload: first),
                    record(type: 0x003C, payload: continuation)]
        }

        var all = header
        for part in parts { all.append(sstPayload(part)) }
        return [record(type: 0x00FC, payload: all)]
    }

    private func sstParts(_ value: String) -> SSTParts {
        let units = Array(value.utf16)
        let compressed = units.allSatisfy { $0 < 0x100 }
        return SSTParts(units: units, flags: compressed ? 0x00 : 0x01,
                        compressed: compressed)
    }

    private func sstHeader(_ parts: SSTParts) -> Data {
        var data = Data(biffBytes(UInt16(parts.units.count)))
        data.append(parts.flags)
        return data
    }

    private func sstPayload(_ parts: SSTParts) -> Data {
        var data = sstHeader(parts)
        data.append(sstChars(parts, start: 0, end: parts.units.count))
        return data
    }

    private func sstChars(_ parts: SSTParts, start: Int, end: Int) -> Data {
        var data = Data()
        if parts.compressed {
            for unit in parts.units[start..<end] { data.append(UInt8(unit)) }
        } else {
            for unit in parts.units[start..<end] {
                data.append(contentsOf: biffBytes(unit))
            }
        }
        return data
    }

    // MARK: 工作表

    private func sheetData(_ sheet: Sheet) -> Data {
        var data = Data()
        data.append(record(type: 0x0809, payload: bofPayload(type: 0x0010)))
        for cell in sheet.cells { data.append(cellRecords(cell)) }
        data.append(record(type: 0x000A, payload: Data()))
        return data
    }

    private func cellRecords(_ cell: Cell) -> Data {
        let row = UInt16(min(max(cell.row, 0), 65_535))
        let column = UInt16(min(max(cell.column, 0), 65_535))

        switch cell.value {
        case .labelSST(let index):
            var payload = Data()
            payload.append(contentsOf: biffBytes(row))
            payload.append(contentsOf: biffBytes(column))
            payload.append(contentsOf: biffBytes(UInt16(0)))
            payload.append(contentsOf: biffBytes(UInt32(index)))
            return record(type: 0x00FD, payload: payload)

        case .label(let text):
            var payload = Data()
            payload.append(contentsOf: biffBytes(row))
            payload.append(contentsOf: biffBytes(column))
            payload.append(contentsOf: biffBytes(UInt16(0)))
            payload.append(xlUnicodeString(text))
            return record(type: 0x0204, payload: payload)

        case .number(let value, let xf):
            var payload = Data()
            payload.append(contentsOf: biffBytes(row))
            payload.append(contentsOf: biffBytes(column))
            payload.append(contentsOf: biffBytes(xf))
            payload.append(contentsOf: biffBytes(value))
            return record(type: 0x0203, payload: payload)

        case .rk(let rk, let xf):
            var payload = Data()
            payload.append(contentsOf: biffBytes(row))
            payload.append(contentsOf: biffBytes(column))
            payload.append(contentsOf: biffBytes(xf))
            payload.append(contentsOf: biffBytes(rk))
            return record(type: 0x027E, payload: payload)

        case .mulrk(let values, let xf):
            guard !values.isEmpty else { return Data() }
            var payload = Data()
            payload.append(contentsOf: biffBytes(row))
            payload.append(contentsOf: biffBytes(column))
            for value in values {
                payload.append(contentsOf: biffBytes(xf))
                payload.append(contentsOf: biffBytes(value))
            }
            payload.append(contentsOf: biffBytes(
                UInt16(min(Int(column) + values.count - 1, 65_535))))
            return record(type: 0x00BD, payload: payload)

        case .formula(let tokens, let result, let string, let xf):
            var payload = Data()
            payload.append(contentsOf: biffBytes(row))
            payload.append(contentsOf: biffBytes(column))
            payload.append(contentsOf: biffBytes(xf))
            var cached = result
            if cached.count < 8 { cached.append(Data(count: 8 - cached.count)) }
            if cached.count > 8 { cached = Data(cached.prefix(8)) }
            payload.append(cached)
            payload.append(contentsOf: biffBytes(UInt16(0)))
            payload.append(contentsOf: biffBytes(UInt32(0)))
            payload.append(contentsOf: biffBytes(UInt16(tokens.count)))
            payload.append(contentsOf: tokens)
            var records = record(type: 0x0006, payload: payload)
            if let string {
                records.append(record(type: 0x0207, payload: xlUnicodeString(string)))
            }
            return records

        case .boolean(let value, let xf):
            var payload = Data()
            payload.append(contentsOf: biffBytes(row))
            payload.append(contentsOf: biffBytes(column))
            payload.append(contentsOf: biffBytes(xf))
            payload.append(value ? 1 : 0)
            payload.append(0)
            return record(type: 0x0205, payload: payload)

        case .error(let code, let xf):
            var payload = Data()
            payload.append(contentsOf: biffBytes(row))
            payload.append(contentsOf: biffBytes(column))
            payload.append(contentsOf: biffBytes(xf))
            payload.append(code)
            payload.append(1)
            return record(type: 0x0205, payload: payload)

        case .blank(let xf):
            var payload = Data()
            payload.append(contentsOf: biffBytes(row))
            payload.append(contentsOf: biffBytes(column))
            payload.append(contentsOf: biffBytes(xf))
            return record(type: 0x0201, payload: payload)
        }
    }

    // MARK: 记录 / 字符串编码

    private func record(type: UInt16, payload: Data) -> Data {
        var data = Data(biffBytes(type))
        data.append(contentsOf: biffBytes(UInt16(payload.count)))
        data.append(payload)
        return data
    }

    private func xlUnicodeString(_ value: String) -> Data {
        let units = Array(value.utf16)
        let compressed = units.allSatisfy { $0 < 0x100 }
        var data = Data(biffBytes(UInt16(units.count)))
        data.append(compressed ? 0x00 : 0x01)
        if compressed {
            for unit in units { data.append(UInt8(unit)) }
        } else {
            for unit in units { data.append(contentsOf: biffBytes(unit)) }
        }
        return data
    }

    private func biffBytes(_ value: UInt16) -> [UInt8] {
        [UInt8(truncatingIfNeeded: value), UInt8(truncatingIfNeeded: value >> 8)]
    }

    private func biffBytes(_ value: UInt32) -> [UInt8] {
        [UInt8(truncatingIfNeeded: value),
         UInt8(truncatingIfNeeded: value >> 8),
         UInt8(truncatingIfNeeded: value >> 16),
         UInt8(truncatingIfNeeded: value >> 24)]
    }

    private func biffBytes(_ value: Double) -> [UInt8] {
        let bits = UInt64(value.bitPattern)
        return [UInt8(truncatingIfNeeded: bits),
                UInt8(truncatingIfNeeded: bits >> 8),
                UInt8(truncatingIfNeeded: bits >> 16),
                UInt8(truncatingIfNeeded: bits >> 24),
                UInt8(truncatingIfNeeded: bits >> 32),
                UInt8(truncatingIfNeeded: bits >> 40),
                UInt8(truncatingIfNeeded: bits >> 48),
                UInt8(truncatingIfNeeded: bits >> 56)]
    }

    // MARK: 测试辅助

    /// RK 编码：bit1 = 整数标志，低 2 位之外存放 30 位有符号整数。
    static func rkInteger(_ value: Int32) -> UInt32 {
        (UInt32(bitPattern: value) << 2) | 0x02
    }

    static func numberResult(_ value: Double) -> Data {
        let bits = UInt64(value.bitPattern)
        var bytes: [UInt8] = []
        for shift in stride(from: 0, through: 56, by: 8) {
            bytes.append(UInt8(truncatingIfNeeded: bits >> UInt64(shift)))
        }
        return Data(bytes)
    }

    /// 公式缓存结果为字符串：0xFFFF + 类型 0（真实文本在后随 STRING 记录）。
    static func stringResult() -> Data {
        Data([0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
    }

    static func areaToken(firstRow: Int, lastRow: Int,
                          firstColumn: Int, lastColumn: Int) -> [UInt8] {
        var bytes: [UInt8] = [0x25]
        bytes += littleEndian(UInt16(firstRow))
        bytes += littleEndian(UInt16(lastRow))
        bytes += littleEndian(UInt16(firstColumn))
        bytes += littleEndian(UInt16(lastColumn))
        return bytes
    }

    static func refToken(row: Int, column: Int) -> [UInt8] {
        var bytes: [UInt8] = [0x24]
        bytes += littleEndian(UInt16(row))
        bytes += littleEndian(UInt16(column))
        return bytes
    }

    static func intToken(_ value: UInt16) -> [UInt8] {
        [0x1E] + littleEndian(value)
    }

    static func funcVarToken(function: UInt16, arguments: UInt8) -> [UInt8] {
        [0x42, arguments] + littleEndian(function)
    }

    static func stringToken(_ value: String) -> [UInt8] {
        let units = Array(value.utf16)
        let compressed = units.allSatisfy { $0 < 0x100 }
        var bytes: [UInt8] = [0x17, UInt8(min(units.count, 255)), compressed ? 0 : 1]
        if compressed {
            bytes += units.prefix(255).map { UInt8($0) }
        } else {
            for unit in units.prefix(255) { bytes += littleEndian(unit) }
        }
        return bytes
    }

    private static func littleEndian(_ value: UInt16) -> [UInt8] {
        [UInt8(truncatingIfNeeded: value), UInt8(truncatingIfNeeded: value >> 8)]
    }
}

// MARK: - PowerPoint 97–2003

/// 仅测试使用：构造带 `PowerPoint Document` 流的最小 `.ppt`。
///
/// 覆盖 `Document` / `SlideListWithText` / `SlidePersistAtom` / `Slide` /
/// `SlideAtom` / `Notes` / `NotesAtom` / 文本 atom。文本记录同时写入
/// SlideListWithText 与 Slide 容器，用于验证顺序优先取文本列表；
/// `includeSlideList` 为 false 时验证回退到文件顺序。
struct PowerPointTestBuilder {

    struct TextBlock {
        let headerType: UInt32
        let text: String
        let useBytesAtom: Bool

        init(headerType: UInt32, text: String, useBytesAtom: Bool = false) {
            self.headerType = headerType
            self.text = text
            self.useBytesAtom = useBytesAtom
        }
    }

    struct Slide {
        let slideId: UInt32
        let blocks: [TextBlock]
        let inSlideList: Bool

        init(slideId: UInt32, blocks: [TextBlock], inSlideList: Bool = true) {
            self.slideId = slideId
            self.blocks = blocks
            self.inSlideList = inSlideList
        }
    }

    struct Notes {
        let slideIdRef: UInt32
        let blocks: [TextBlock]

        init(slideIdRef: UInt32, blocks: [TextBlock]) {
            self.slideIdRef = slideIdRef
            self.blocks = blocks
        }
    }

    var slides: [Slide]
    var notes: [Notes]
    var includeDocument: Bool
    var includeSlideList: Bool
    /// 非空时只写该类型的一条顶层记录，用于模拟 PowerPoint 95 / 损坏文件。
    var firstRecordType: UInt16?
    var firstRecordVersion: UInt8
    var encrypted: Bool
    var streamName: String
    /// 在 Document 内部再包 N 层容器，用于测试记录深度上限。
    var extraNesting: Int
    /// 末尾追加 N 条零长度记录，用于测试记录总数上限。
    var trailingRecords: Int
    /// OLE2 扇区大小；大量记录时用 4096 以避免 FAT 扇区超过头部 DIFAT 容量。
    var sectorSize: Int

    init(slides: [Slide] = [],
         notes: [Notes] = [],
         includeDocument: Bool = true,
         includeSlideList: Bool = true,
         firstRecordType: UInt16? = nil,
         firstRecordVersion: UInt8 = 0,
         encrypted: Bool = false,
         streamName: String = "PowerPoint Document",
         extraNesting: Int = 0,
         trailingRecords: Int = 0,
         sectorSize: Int = 512) {
        self.slides = slides
        self.notes = notes
        self.includeDocument = includeDocument
        self.includeSlideList = includeSlideList
        self.firstRecordType = firstRecordType
        self.firstRecordVersion = firstRecordVersion
        self.encrypted = encrypted
        self.streamName = streamName
        self.extraNesting = extraNesting
        self.trailingRecords = trailingRecords
        self.sectorSize = sectorSize
    }

    func build() throws -> OLE2TestFile {
        var content = Data()
        if let firstRecordType {
            content.append(Self.record(type: firstRecordType,
                                       version: firstRecordVersion))
        } else {
            if includeDocument { content.append(documentRecord()) }
            for slide in slides { content.append(slideRecord(slide)) }
            for note in notes { content.append(notesRecord(note)) }
        }
        for _ in 0..<trailingRecords {
            content.append(Self.record(type: 0x0000, payload: Data()))
        }

        var streams: [OLE2TestBuilder.StreamSpec] = [
            .init(streamName, content),
        ]
        if encrypted {
            streams.append(.init("EncryptionInfo", Data(repeating: 0, count: 64)))
        }
        return try OLE2TestBuilder(sectorSize: sectorSize, streams: streams).build()
    }

    // MARK: 文档结构

    private func documentRecord() -> Data {
        var payload = Data()
        payload.append(Self.record(type: 0x03E9, payload: Data(repeating: 0, count: 8)))

        if includeSlideList {
            let listed = slides.filter(\.inSlideList)
            if !listed.isEmpty {
                var list = Data()
                for slide in listed {
                    list.append(Self.slidePersistAtom(slideId: slide.slideId,
                                                      textCount: UInt32(slide.blocks.count)))
                    for block in slide.blocks { list.append(textRecords(block)) }
                }
                payload.append(Self.container(type: 0x0FF0, instance: 0, payload: list))
            }
        }

        for _ in 0..<extraNesting {
            payload = Self.container(type: 0x0FFF, payload: payload)
        }
        return Self.container(type: 0x03E8, payload: payload)
    }

    private func slideRecord(_ slide: Slide) -> Data {
        var payload = Data()
        var atom = Data(repeating: 0, count: 16)
        Self.writeUInt32(slide.slideId, into: &atom, at: 8)
        payload.append(Self.record(type: 0x03EF, payload: atom))
        for block in slide.blocks {
            payload.append(Self.container(type: 0xF004, payload: textRecords(block)))
        }
        return Self.container(type: 0x03EE, payload: payload)
    }

    private func notesRecord(_ note: Notes) -> Data {
        var atom = Data(repeating: 0, count: 8)
        Self.writeUInt32(note.slideIdRef, into: &atom, at: 0)
        var payload = Self.record(type: 0x03F1, payload: atom)
        for block in note.blocks { payload.append(textRecords(block)) }
        return Self.container(type: 0x03F0, payload: payload)
    }

    private func textRecords(_ block: TextBlock) -> Data {
        var data = Data()
        data.append(Self.record(type: 0x0F9F,
                                payload: Data(Self.u32(block.headerType))))
        if block.useBytesAtom {
            let bytes = block.text.data(using: .windowsCP1252) ?? Data(block.text.utf8)
            data.append(Self.record(type: 0x0FA8, payload: bytes))
        } else {
            var payload = Data()
            for unit in block.text.utf16 {
                payload.append(contentsOf: Self.u16(unit))
            }
            data.append(Self.record(type: 0x0FA0, payload: payload))
        }
        return data
    }

    // MARK: 记录编码

    private static func record(type: UInt16, version: UInt8 = 0,
                               instance: UInt16 = 0, payload: Data = Data()) -> Data {
        var data = Data(u16((instance << 4) | UInt16(version & 0x0F)))
        data.append(contentsOf: u16(type))
        data.append(contentsOf: u32(UInt32(payload.count)))
        data.append(payload)
        return data
    }

    private static func container(type: UInt16, instance: UInt16 = 0,
                                  payload: Data) -> Data {
        record(type: type, version: 0x0F, instance: instance, payload: payload)
    }

    private static func slidePersistAtom(slideId: UInt32, textCount: UInt32) -> Data {
        var payload = Data(repeating: 0, count: 20)
        writeUInt32(slideId, into: &payload, at: 0)
        writeUInt32(textCount, into: &payload, at: 8)
        writeUInt32(slideId, into: &payload, at: 12)
        return record(type: 0x03F3, payload: payload)
    }

    private static func writeUInt32(_ value: UInt32, into data: inout Data, at offset: Int) {
        let bytes = u32(value)
        data.replaceSubrange(offset..<(offset + 4), with: bytes)
    }

    private static func u16(_ value: UInt16) -> [UInt8] {
        [UInt8(truncatingIfNeeded: value), UInt8(truncatingIfNeeded: value >> 8)]
    }

    private static func u32(_ value: UInt32) -> [UInt8] {
        [UInt8(truncatingIfNeeded: value),
         UInt8(truncatingIfNeeded: value >> 8),
         UInt8(truncatingIfNeeded: value >> 16),
         UInt8(truncatingIfNeeded: value >> 24)]
    }
}

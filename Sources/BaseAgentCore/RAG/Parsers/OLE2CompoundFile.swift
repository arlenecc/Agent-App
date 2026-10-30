import Foundation

/// OLE2 复合文档读取出错。上层统一定义解析错误文案，这里只描述容器问题。
enum OLE2Error: LocalizedError {
    case invalid(String)

    var errorDescription: String? {
        switch self {
        case .invalid(let message): return message
        }
    }
}

/// 只读 OLE2（Compound File Binary Format）容器。
///
/// 支持 Word/Excel/PowerPoint 97–2003 使用的 512 与 4096 字节扇区、
/// DIFAT/FAT/miniFAT、目录项与大小写不敏感的流名。这里不解析任何
/// 应用层记录，只负责把命名流安全地取出来。
struct OLE2CompoundFile: Sendable {

    struct Entry: Sendable, Equatable {
        let name: String
        /// 0 = 空项，1 = storage，2 = stream，5 = root。
        let type: UInt8
        let startSector: UInt32
        let size: UInt64
    }

    private static let signature: [UInt8] = [
        0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1,
    ]
    private static let headerSize = 512
    private static let freeSector: UInt32 = 0xFFFF_FFFF
    private static let endOfChain: UInt32 = 0xFFFF_FFFE
    private static let maxRegularSector: UInt32 = 0xFFFF_FFFA
    private static let miniStreamCutoffDefault: UInt32 = 4096
    private static let miniSectorSizeDefault: UInt16 = 6

    let data: Data
    let majorVersion: UInt16
    let sectorSize: Int
    let miniSectorSize: Int
    let miniStreamCutoff: Int
    let entries: [Entry]

    private let fat: [UInt32]
    private let miniFat: [UInt32]
    private let miniStream: Data
    private let sectorCount: Int

    init(data: Data) throws {
        guard data.count >= Self.headerSize else {
            throw OLE2Error.invalid("OLE2 文件头不完整")
        }
        guard data.count <= LegacyOfficeLimits.maxFileSize else {
            throw OLE2Error.invalid("文件超过 \(LegacyOfficeLimits.maxFileSize / 1024 / 1024)MB 上限")
        }
        guard Array(data[0..<8]) == Self.signature else {
            throw OLE2Error.invalid("不是有效的 OLE2 复合文档")
        }

        let byteOrder = try Self.u16(data, 0x1C)
        guard byteOrder == 0xFFFE else {
            throw OLE2Error.invalid("OLE2 字节序标记无效")
        }

        let major = try Self.u16(data, 0x1A)
        let sectorShift = try Self.u16(data, 0x1E)
        let miniShift = try Self.u16(data, 0x20)
        guard major == 3 || major == 4 else {
            throw OLE2Error.invalid("不支持的 OLE2 主版本 \(major)")
        }
        guard (major == 3 && sectorShift == 9) || (major == 4 && sectorShift == 12) else {
            throw OLE2Error.invalid("OLE2 扇区大小与版本不匹配")
        }
        guard miniShift == Self.miniSectorSizeDefault else {
            throw OLE2Error.invalid("OLE2 mini 扇区大小无效")
        }

        let sectorSize = 1 << Int(sectorShift)
        let miniSectorSize = 1 << Int(miniShift)
        let cutoffRaw = try Self.u32(data, 0x38)
        let cutoff = Int(cutoffRaw == 0 ? Self.miniStreamCutoffDefault : cutoffRaw)
        guard cutoff >= miniSectorSize, cutoff <= LegacyOfficeLimits.maxStreamSize else {
            throw OLE2Error.invalid("OLE2 mini stream 阈值无效")
        }

        let availableSectors = (data.count - Self.headerSize) / sectorSize
        guard availableSectors > 0 else {
            throw OLE2Error.invalid("OLE2 文件没有扇区数据")
        }
        let sectorCount = availableSectors

        let directoryStart = try Self.u32(data, 0x30)
        let fatSectorCount = Int(try Self.u32(data, 0x2C))
        let miniFatStart = try Self.u32(data, 0x3C)
        let miniFatSectorCount = Int(try Self.u32(data, 0x40))
        let difatStart = try Self.u32(data, 0x44)
        let difatSectorCount = Int(try Self.u32(data, 0x48))

        let entriesPerSector = sectorSize / 4
        // 每个 FAT 扇区覆盖 entriesPerSector 个扇区；多允许一个余量以兼容
        // 写入器多分配的冗余 FAT，但拒绝恶意放大的 FAT 表。
        let maxFatSectors = (sectorCount + entriesPerSector - 1) / entriesPerSector + 1
        guard fatSectorCount > 0, fatSectorCount <= maxFatSectors else {
            throw OLE2Error.invalid("OLE2 FAT 扇区数量异常")
        }

        // 1. 收集 FAT 扇区编号：头部 109 个，其余由 DIFAT 链提供。
        var fatSectorIDs: [UInt32] = []
        fatSectorIDs.reserveCapacity(fatSectorCount)
        let headerDIFATEntries = 109
        for index in 0..<min(fatSectorCount, headerDIFATEntries) {
            let sector = try Self.u32(data, 0x4C + index * 4)
            guard sector < Self.maxRegularSector else {
                throw OLE2Error.invalid("OLE2 DIFAT 表项无效")
            }
            fatSectorIDs.append(sector)
        }

        if fatSectorIDs.count < fatSectorCount {
            guard difatSectorCount > 0, difatStart != Self.endOfChain else {
                throw OLE2Error.invalid("OLE2 FAT 表被截断")
            }
            var current = difatStart
            var visited = Set<UInt32>()
            let perSector = entriesPerSector - 1
            var visitedCount = 0
            while fatSectorIDs.count < fatSectorCount {
                guard current != Self.endOfChain else { break }
                guard current < Self.maxRegularSector else {
                    throw OLE2Error.invalid("OLE2 DIFAT 扇区编号无效")
                }
                guard visited.insert(current).inserted else {
                    throw OLE2Error.invalid("OLE2 DIFAT 链存在循环")
                }
                visitedCount += 1
                guard visitedCount <= difatSectorCount else {
                    throw OLE2Error.invalid("OLE2 DIFAT 链超过声明长度")
                }
                let range = try Self.sectorRange(id: current,
                                                 sectorSize: sectorSize,
                                                 dataSize: data.count)
                for index in 0..<perSector where fatSectorIDs.count < fatSectorCount {
                    let sector = try Self.u32(data, range.lowerBound + index * 4)
                    if sector == Self.freeSector { continue }
                    guard sector < Self.maxRegularSector else {
                        throw OLE2Error.invalid("OLE2 DIFAT 表项无效")
                    }
                    fatSectorIDs.append(sector)
                }
                current = try Self.u32(data, range.lowerBound + perSector * 4)
            }
        }
        guard fatSectorIDs.count == fatSectorCount else {
            throw OLE2Error.invalid("OLE2 FAT 表被截断")
        }
        var uniqueFATSectors = Set<UInt32>()
        for sector in fatSectorIDs {
            guard uniqueFATSectors.insert(sector).inserted else {
                throw OLE2Error.invalid("OLE2 FAT 扇区重复")
            }
        }

        // 2. 读取 FAT 表本身。
        var fat: [UInt32] = []
        fat.reserveCapacity(fatSectorCount * entriesPerSector)
        for sector in fatSectorIDs {
            let range = try Self.sectorRange(id: sector,
                                             sectorSize: sectorSize,
                                             dataSize: data.count)
            fat.append(contentsOf: try Self.u32Array(data, start: range.lowerBound,
                                                     count: entriesPerSector))
        }

        // 3. 读取目录链并解析目录项。
        let declaredDirectorySectors = major >= 4 ? Int(try Self.u32(data, 0x28)) : 0
        let directorySectorLimit = declaredDirectorySectors > 0
            ? declaredDirectorySectors : sectorCount
        let directorySectors = try Self.chain(start: directoryStart,
                                              table: fat,
                                              limit: directorySectorLimit,
                                              label: "目录")
        guard !directorySectors.isEmpty else {
            throw OLE2Error.invalid("OLE2 目录链为空")
        }
        if declaredDirectorySectors > 0, directorySectors.count != declaredDirectorySectors {
            throw OLE2Error.invalid("OLE2 目录链被截断")
        }
        var directoryData = Data()
        directoryData.reserveCapacity(directorySectors.count * sectorSize)
        for sector in directorySectors {
            let range = try Self.sectorRange(id: sector,
                                             sectorSize: sectorSize,
                                             dataSize: data.count)
            directoryData.append(data[range])
        }
        let entries = try Self.parseDirectory(directoryData)
        guard entries.contains(where: { $0.type == 5 }) else {
            throw OLE2Error.invalid("OLE2 目录缺少根项")
        }
        guard let root = entries.first(where: { $0.type == 5 }) else {
            throw OLE2Error.invalid("OLE2 目录缺少根项")
        }

        // 4. 读取 miniFAT。
        var miniFat: [UInt32] = []
        if miniFatStart == Self.endOfChain, miniFatSectorCount > 0 {
            throw OLE2Error.invalid("OLE2 miniFAT 声明了扇区但缺少起始位置")
        }
        if miniFatStart != Self.endOfChain {
            guard miniFatStart < Self.maxRegularSector else {
                throw OLE2Error.invalid("OLE2 miniFAT 链无效")
            }
            let declared = miniFatSectorCount
            let limit = declared > 0 ? declared : sectorCount
            let sectors = try Self.chain(start: miniFatStart,
                                         table: fat,
                                         limit: limit,
                                         label: "miniFAT")
            if declared > 0, sectors.count != declared {
                throw OLE2Error.invalid("OLE2 miniFAT 链被截断")
            }
            var miniFatBytes = Data()
            miniFatBytes.reserveCapacity(sectors.count * sectorSize)
            for sector in sectors {
                let range = try Self.sectorRange(id: sector,
                                                 sectorSize: sectorSize,
                                                 dataSize: data.count)
                miniFatBytes.append(data[range])
            }
            miniFat = try Self.u32Array(miniFatBytes, start: 0,
                                        count: miniFatBytes.count / 4)
        }

        // 5. 读取 mini stream 本体（根项目的普通 FAT 链）。
        var miniStream = Data()
        if root.size > 0 {
            guard root.size <= UInt64(LegacyOfficeLimits.maxStreamSize) else {
                throw OLE2Error.invalid("OLE2 mini stream 超过单流上限")
            }
            let needed = Int((root.size + UInt64(sectorSize) - 1) / UInt64(sectorSize))
            let sectors = try Self.chain(start: root.startSector,
                                         table: fat,
                                         limit: needed,
                                         label: "mini stream")
            guard sectors.count == needed else {
                throw OLE2Error.invalid("OLE2 mini stream 链被截断")
            }
            miniStream.reserveCapacity(needed * sectorSize)
            for sector in sectors {
                let range = try Self.sectorRange(id: sector,
                                                 sectorSize: sectorSize,
                                                 dataSize: data.count)
                miniStream.append(data[range])
            }
            if miniStream.count > Int(root.size) {
                miniStream = Data(miniStream.prefix(Int(root.size)))
            }
        }

        self.data = data
        self.majorVersion = major
        self.sectorSize = sectorSize
        self.miniSectorSize = miniSectorSize
        self.miniStreamCutoff = cutoff
        self.entries = entries
        self.fat = fat
        self.miniFat = miniFat
        self.miniStream = miniStream
        self.sectorCount = sectorCount
    }

    static func open(url: URL) throws -> OLE2CompoundFile {
        let values = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        guard values.isRegularFile != false else {
            throw OLE2Error.invalid("不是普通文件：\(url.lastPathComponent)")
        }
        if let size = values.fileSize, size > LegacyOfficeLimits.maxFileSize {
            throw OLE2Error.invalid("文件超过 \(LegacyOfficeLimits.maxFileSize / 1024 / 1024)MB 上限")
        }
        let data = try Data(contentsOf: url, options: [.mappedIfSafe])
        return try OLE2CompoundFile(data: data)
    }

    /// 目录中所有命名流（不含 storage / root）。
    var streamNames: [String] {
        entries.filter { $0.type == 2 }.map(\.name)
    }

    func containsStream(named name: String) -> Bool {
        entries.contains { $0.type == 2 && $0.name.caseInsensitiveCompare(name) == .orderedSame }
    }

    func streamData(named name: String) throws -> Data {
        guard let entry = entries.first(where: {
            $0.type == 2 && $0.name.caseInsensitiveCompare(name) == .orderedSame
        }) else {
            throw OLE2Error.invalid("缺少流：\(name)")
        }
        guard entry.size <= UInt64(LegacyOfficeLimits.maxStreamSize) else {
            throw OLE2Error.invalid("流 \(entry.name) 超过单流上限")
        }
        guard entry.size > 0 else { return Data() }

        if entry.size < UInt64(miniStreamCutoff) {
            return try miniStreamData(entry)
        }
        return try regularStreamData(entry)
    }

    // MARK: - 流读取

    private func regularStreamData(_ entry: Entry) throws -> Data {
        let needed = Int((entry.size + UInt64(sectorSize) - 1) / UInt64(sectorSize))
        let sectors = try Self.chain(start: entry.startSector,
                                     table: fat,
                                     limit: needed,
                                     label: "流 \(entry.name)")
        guard sectors.count == needed else {
            throw OLE2Error.invalid("流 \(entry.name) 的扇区链被截断")
        }
        var result = Data()
        result.reserveCapacity(needed * sectorSize)
        for sector in sectors {
            let range = try Self.sectorRange(id: sector,
                                             sectorSize: sectorSize,
                                             dataSize: data.count)
            result.append(data[range])
        }
        if result.count > Int(entry.size) {
            result = Data(result.prefix(Int(entry.size)))
        }
        return result
    }

    private func miniStreamData(_ entry: Entry) throws -> Data {
        guard !miniFat.isEmpty else {
            throw OLE2Error.invalid("流 \(entry.name) 需要 miniFAT，但文件缺少 miniFAT")
        }
        let needed = Int((entry.size + UInt64(miniSectorSize) - 1) / UInt64(miniSectorSize))
        let sectors = try Self.chain(start: entry.startSector,
                                     table: miniFat,
                                     limit: needed,
                                     label: "mini 流 \(entry.name)")
        guard sectors.count == needed else {
            throw OLE2Error.invalid("mini 流 \(entry.name) 的扇区链被截断")
        }
        var result = Data()
        result.reserveCapacity(needed * miniSectorSize)
        for sector in sectors {
            let offset = Int(sector) * miniSectorSize
            guard offset <= miniStream.count - miniSectorSize else {
                throw OLE2Error.invalid("mini 流 \(entry.name) 引用了 mini stream 之外的扇区")
            }
            result.append(miniStream[offset..<(offset + miniSectorSize)])
        }
        if result.count > Int(entry.size) {
            result = Data(result.prefix(Int(entry.size)))
        }
        return result
    }

    // MARK: - 头部 / FAT / 目录

    private static func sectorRange(id: UInt32, sectorSize: Int, dataSize: Int) throws -> Range<Int> {
        guard id < maxRegularSector else {
            throw OLE2Error.invalid("OLE2 扇区编号无效：\(id)")
        }
        let start = UInt64(headerSize) + UInt64(id) * UInt64(sectorSize)
        let end = start + UInt64(sectorSize)
        guard start <= UInt64(dataSize), end <= UInt64(dataSize) else {
            throw OLE2Error.invalid("OLE2 扇区超出文件范围")
        }
        return Int(start)..<Int(end)
    }

    /// 沿 FAT/miniFAT 链收集扇区；最多取 `limit` 个，检测循环与越界。
    private static func chain(start: UInt32, table: [UInt32], limit: Int,
                              label: String) throws -> [UInt32] {
        guard start != endOfChain else { return [] }
        guard start < maxRegularSector else {
            throw OLE2Error.invalid("\(label) 起始扇区无效")
        }
        guard limit > 0 else { return [] }
        var result: [UInt32] = []
        result.reserveCapacity(min(limit, table.count))
        var visited = Set<UInt32>()
        var current = start
        while current != endOfChain, result.count < limit {
            guard current < maxRegularSector else {
                throw OLE2Error.invalid("\(label) 链包含非法扇区")
            }
            guard visited.insert(current).inserted else {
                throw OLE2Error.invalid("\(label) 链存在循环")
            }
            guard Int(current) < table.count else {
                throw OLE2Error.invalid("\(label) 链引用了 FAT 表之外的扇区")
            }
            result.append(current)
            current = table[Int(current)]
        }
        return result
    }

    private static func parseDirectory(_ directory: Data) throws -> [Entry] {
        guard directory.count >= 128 else {
            throw OLE2Error.invalid("OLE2 目录数据不完整")
        }
        var result: [Entry] = []
        var offset = 0
        while offset <= directory.count - 128 {
            let type = directory[offset + 0x42]
            if type != 0 {
                let nameLength = Int(try u16(directory, offset + 0x40))
                let rawNameBytes = min(max(nameLength - 2, 0), 64)
                let evenBytes = rawNameBytes - (rawNameBytes % 2)
                var name = ""
                if evenBytes >= 2 {
                    let range = (offset)..<(offset + evenBytes)
                    if let decoded = String(data: Data(directory[range]),
                                            encoding: .utf16LittleEndian) {
                        name = decoded.replacingOccurrences(of: "\u{0000}", with: "")
                    }
                }
                let startSector = try u32(directory, offset + 0x74)
                let size = try u64(directory, offset + 0x78)
                result.append(Entry(name: name, type: type,
                                    startSector: startSector, size: size))
            }
            offset += 128
        }
        return result
    }

    private static func u16(_ data: Data, _ offset: Int) throws -> UInt16 {
        guard offset >= 0, offset <= data.count - 2 else {
            throw OLE2Error.invalid("OLE2 数据被截断")
        }
        return data.withUnsafeBytes {
            UInt16(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt16.self))
        }
    }

    private static func u32(_ data: Data, _ offset: Int) throws -> UInt32 {
        guard offset >= 0, offset <= data.count - 4 else {
            throw OLE2Error.invalid("OLE2 数据被截断")
        }
        return data.withUnsafeBytes {
            UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
        }
    }

    private static func u64(_ data: Data, _ offset: Int) throws -> UInt64 {
        guard offset >= 0, offset <= data.count - 8 else {
            throw OLE2Error.invalid("OLE2 数据被截断")
        }
        return data.withUnsafeBytes {
            UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt64.self))
        }
    }

    private static func u32Array(_ data: Data, start: Int, count: Int) throws -> [UInt32] {
        guard count >= 0, start >= 0,
              start <= data.count,
              count <= (data.count - start) / 4 else {
            throw OLE2Error.invalid("OLE2 32 位表被截断")
        }
        var result: [UInt32] = []
        result.reserveCapacity(count)
        for index in 0..<count {
            result.append(try u32(data, start + index * 4))
        }
        return result
    }
}

#include "CPDFShim.h"
#include <string.h>
#include <math.h>
#include <stdbool.h>

static bool dict_name_equals(CGPDFDictionaryRef dict, const char *key, const char *expected) {
    const char *name = NULL;
    return CGPDFDictionaryGetName(dict, key, &name) && name
        && strcmp(name, expected) == 0;
}

static bool filter_contains_dct(CGPDFDictionaryRef dict) {
    if (dict_name_equals(dict, "Filter", "DCTDecode")) return true;
    CGPDFArrayRef array = NULL;
    if (!CGPDFDictionaryGetArray(dict, "Filter", &array) || !array) return false;
    __block bool found = false;
    CGPDFArrayApplyBlock(array, ^bool (size_t index, CGPDFObjectRef value, void *info) {
        const char *name = NULL;
        if (CGPDFObjectGetValue(value, kCGPDFObjectTypeString, &name) && name
            && strcmp(name, "DCTDecode") == 0) {
            found = true;
            return false;
        }
        return true;
    }, NULL);
    return found;
}

typedef struct {
    CGPDFStreamRef streams[8];
    int count;
} StreamCollector;

CFDataRef CPDFShimCopyEmbeddedPageJPEG(CGPDFPageRef page) {
    if (!page) return NULL;
    if (((CGPDFPageGetRotationAngle(page) % 360) + 360) % 360 != 0) return NULL;

    CGPDFDictionaryRef pageDict = CGPDFPageGetDictionary(page);
    if (!pageDict) return NULL;
    CGPDFDictionaryRef resources = NULL;
    if (!CGPDFDictionaryGetDictionary(pageDict, "Resources", &resources) || !resources) return NULL;
    CGPDFDictionaryRef xobjects = NULL;
    if (!CGPDFDictionaryGetDictionary(resources, "XObject", &xobjects) || !xobjects) return NULL;

    // 收集本页全部流对象（正常扫描页只有 1 张整页图；防御异常文件上限 8）。
    StreamCollector collector;
    memset(&collector, 0, sizeof(collector));
    CGPDFDictionaryApplyBlock(xobjects,
        ^bool (const char *key, CGPDFObjectRef value, void *info) {
        StreamCollector *c = (StreamCollector *)info;
        CGPDFStreamRef stream = NULL;
        if (CGPDFObjectGetValue(value, kCGPDFObjectTypeStream, &stream) && stream
            && c->count < 8) {
            c->streams[c->count++] = stream;
        }
        return true;
    }, &collector);
    if (collector.count != 1) return NULL;

    CGPDFStreamRef stream = collector.streams[0];
    CGPDFDictionaryRef streamDict = CGPDFStreamGetDictionary(stream);
    if (!streamDict) return NULL;

    if (!dict_name_equals(streamDict, "Subtype", "Image")) return NULL;
    if (!filter_contains_dct(streamDict)) return NULL;

    CGPDFObjectRef object = NULL;
    if (CGPDFDictionaryGetObject(streamDict, "SMask", &object)) return NULL;
    if (CGPDFDictionaryGetObject(streamDict, "Mask", &object)) return NULL;

    CGPDFInteger width = 0, height = 0, bitsPerComponent = 0;
    if (!CGPDFDictionaryGetInteger(streamDict, "Width", &width)
        || !CGPDFDictionaryGetInteger(streamDict, "Height", &height)
        || width < 64 || height < 64
        || !CGPDFDictionaryGetInteger(streamDict, "BitsPerComponent", &bitsPerComponent)
        || bitsPerComponent != 8) return NULL;

    // 与 Swift 侧 PDFPageRenderer.maxEmbeddedPixelCount 保持一致：
    // 超高像素图片放弃直提取，交给有界栅格化（目标长边 1408/1500）。
    // 否则 ImageIO 解码器仍得面对数亿像素，单页就可能要几 GB 内存。
    if (width > 20000 || height > 20000
        || (double)width * (double)height > 200000000.0) return NULL;

    CGRect box = CGPDFPageGetBoxRect(page, kCGPDFCropBox);
    if (box.size.width > 0 && box.size.height > 0) {
        double pageAR = (double)box.size.width / (double)box.size.height;
        double imageAR = (double)width / (double)height;
        if (pageAR > 0 && fabs(pageAR - imageAR) / pageAR > 0.25) return NULL;
    }

    CGPDFDataFormat format = CGPDFDataFormatRaw;
    CFDataRef data = CGPDFStreamCopyData(stream, &format);
    if (!data) return NULL;
    if (format != CGPDFDataFormatJPEGEncoded) {
        CFRelease(data);
        return NULL;
    }
    return data;
}

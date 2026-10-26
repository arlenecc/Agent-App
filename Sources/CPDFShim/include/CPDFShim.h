#ifndef CPDFSHIM_H
#define CPDFSHIM_H
#include <CoreGraphics/CoreGraphics.h>

/// 提取扫描页唯一的 DCTDecode（JPEG）整页图像，跳过栅格化。
///
/// 全部条件满足才返回（否则 NULL，调用方回退 CoreGraphics 渲染）：
/// - 页面旋转角为 0（/Rotate 需要重排位图，直接走渲染路径更稳）；
/// - 页面 XObject 中图像流恰好 1 张（多图页面不启发式拼合）；
/// - Subtype == Image，Filter 含 DCTDecode（单名或数组）；
/// - 无 SMask / Mask（带蒙版需与页面合成）；
/// - BitsPerComponent == 8，宽高 ≥ 64；
/// - 图像长宽比与页面裁剪框差 ≤ 25%（过滤插图误判为整页图）。
///
/// @return 满足条件时返回 retained CFDataRef（JPEG 字节流，调用方负责
///         CFRelease），否则 NULL。
CFDataRef CPDFShimCopyEmbeddedPageJPEG(CGPDFPageRef page);

#endif /* CPDFSHIM_H */

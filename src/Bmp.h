#pragma once
//
// Bmp.h  --  write uncompressed 24-bit BMP images.
//
// BMP is about the simplest image format there is: two fixed headers
// followed by raw pixel bytes, no compression.  Three quirks to respect:
//
//   1. Rows are stored bottom-to-top (the last image row comes first in
//      the file) when the header height is positive.
//   2. Pixels are stored B, G, R -- not R, G, B.
//   3. Each pixel row is padded with zero bytes up to a multiple of 4.
//
// We always write 24-bit colour (3 bytes/pixel).  Greyscale is just
// R = G = B, which keeps the format uniform and leaves the door open
// for colour later (e.g. a colormap for spectrogram images).
//
// Usage:
//     Bitmap bmp(width, height);              // black canvas
//     bmp.set_gray(x, y, 200);                // one grey pixel
//     bmp.set_pixel(x, y, 255, 0, 0);         // one red pixel
//     save_bmp("out.bmp", bmp);
//
// Coordinates are top-left origin: (0,0) is the top-left pixel, x grows
// right, y grows down.  The bottom-to-top file ordering is handled
// internally by save_bmp, so callers never think about it.
//
#include "Endian.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace weft {

class Bitmap {
public:
    Bitmap(int width, int height)
        : w_(width), h_(height),
          rgb_(static_cast<std::size_t>(width) * height * 3, 0)
    {
        if (width <= 0 || height <= 0)
            throw std::invalid_argument("Bitmap: dimensions must be positive");
    }

    int width()  const { return w_; }
    int height() const { return h_; }

    void set_pixel(int x, int y, std::uint8_t r, std::uint8_t g, std::uint8_t b) {
        if (x < 0 || x >= w_ || y < 0 || y >= h_) return;   // clip silently
        const std::size_t i = (static_cast<std::size_t>(y) * w_ + x) * 3;
        rgb_[i + 0] = r;
        rgb_[i + 1] = g;
        rgb_[i + 2] = b;
    }

    void set_gray(int x, int y, std::uint8_t v) { set_pixel(x, y, v, v, v); }

    // Row-major top-to-bottom, RGB triplets.  save_bmp reorders for the file.
    const std::vector<std::uint8_t>& data() const { return rgb_; }

private:
    int w_, h_;
    std::vector<std::uint8_t> rgb_;
};

inline void save_bmp(const std::string& path, const Bitmap& bmp) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Bmp: cannot open for write: " + path);

    const int w = bmp.width();
    const int h = bmp.height();

    // Each row: w*3 pixel bytes, padded up to a multiple of 4.
    const std::uint32_t row_bytes  = static_cast<std::uint32_t>(w) * 3;
    const std::uint32_t padding    = (4 - (row_bytes % 4)) % 4;
    const std::uint32_t padded_row = row_bytes + padding;
    const std::uint32_t pixel_size = padded_row * static_cast<std::uint32_t>(h);

    const std::uint32_t file_header = 14;
    const std::uint32_t info_header = 40;
    const std::uint32_t data_offset = file_header + info_header;
    const std::uint32_t file_size   = data_offset + pixel_size;

    // ---- BITMAPFILEHEADER (14 bytes) ----
    out.write("BM", 2);
    detail::write_le_u32(out, file_size);
    detail::write_le_u16(out, 0);            // reserved1
    detail::write_le_u16(out, 0);            // reserved2
    detail::write_le_u32(out, data_offset);

    // ---- BITMAPINFOHEADER (40 bytes) ----
    detail::write_le_u32(out, info_header);
    detail::write_le_u32(out, static_cast<std::uint32_t>(w));
    detail::write_le_u32(out, static_cast<std::uint32_t>(h));   // +ve => bottom-up
    detail::write_le_u16(out, 1);            // colour planes
    detail::write_le_u16(out, 24);           // bits per pixel
    detail::write_le_u32(out, 0);            // BI_RGB, no compression
    detail::write_le_u32(out, pixel_size);   // image size
    detail::write_le_u32(out, 2835);         // x px/metre (~72 dpi)
    detail::write_le_u32(out, 2835);         // y px/metre
    detail::write_le_u32(out, 0);            // colours in palette
    detail::write_le_u32(out, 0);            // "important" colours

    // ---- Pixel data: bottom row first, BGR order, zero-padded rows ----
    const auto& rgb = bmp.data();
    const unsigned char pad[3] = {0, 0, 0};
    for (int y = h - 1; y >= 0; --y) {
        for (int x = 0; x < w; ++x) {
            const std::size_t i = (static_cast<std::size_t>(y) * w + x) * 3;
            unsigned char bgr[3] = { rgb[i + 2], rgb[i + 1], rgb[i + 0] };
            out.write(reinterpret_cast<char*>(bgr), 3);
        }
        if (padding) out.write(reinterpret_cast<const char*>(pad), padding);
    }

    if (!out) throw std::runtime_error("Bmp: write failed: " + path);
}

} // namespace weft

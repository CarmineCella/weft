// test_bmp.cpp
//
// Tests for the BMP writer.  No reader exists, so we write a small image
// and verify the raw file bytes: header fields and a few pixels, checking
// the bottom-to-top row order, BGR channel order, and 4-byte row padding.
//
#include "Bmp.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using weft::Bitmap;
using weft::save_bmp;

static int g_run = 0, g_failed = 0;
static void check(bool cond, const std::string& name) {
    ++g_run;
    std::cout << (cond ? "  [ ok ] " : "  [FAIL] ") << name << '\n';
    if (!cond) ++g_failed;
}

static std::uint16_t rd_u16(const std::vector<unsigned char>& b, std::size_t o) {
    return std::uint16_t(b[o]) | (std::uint16_t(b[o + 1]) << 8);
}
static std::uint32_t rd_u32(const std::vector<unsigned char>& b, std::size_t o) {
    return  std::uint32_t(b[o])
         | (std::uint32_t(b[o + 1]) <<  8)
         | (std::uint32_t(b[o + 2]) << 16)
         | (std::uint32_t(b[o + 3]) << 24);
}

int main() {
    std::cout << "weft :: Bmp tests\n";

    const std::string path = "/tmp/weft_test.bmp";

    // 3x2 image.  Row bytes = 9, padded to 12 (3 pad bytes/row).
    Bitmap bmp(3, 2);
    bmp.set_pixel(0, 0, 255,   0,   0);   // red
    bmp.set_pixel(1, 0,   0, 255,   0);   // green
    bmp.set_pixel(2, 0,   0,   0, 255);   // blue
    bmp.set_pixel(0, 1, 255, 255, 255);   // white
    bmp.set_pixel(1, 1,   0,   0,   0);   // black
    bmp.set_gray (2, 1, 128);             // grey

    save_bmp(path, bmp);

    std::ifstream in(path, std::ios::binary);
    std::vector<unsigned char> f((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());
    in.close();
    std::remove(path.c_str());

    // ---- Header ----
    check(f.size() == 78,                         "file size is 78 bytes");
    check(f[0] == 'B' && f[1] == 'M',             "magic is 'BM'");
    check(rd_u32(f, 2)  == 78,                    "header file_size = 78");
    check(rd_u32(f, 10) == 54,                    "data offset = 54");
    check(rd_u32(f, 14) == 40,                    "info header size = 40");
    check(rd_u32(f, 18) == 3,                     "width = 3");
    check(rd_u32(f, 22) == 2,                     "height = 2");
    check(rd_u16(f, 26) == 1,                     "planes = 1");
    check(rd_u16(f, 28) == 24,                    "bpp = 24");
    check(rd_u32(f, 30) == 0,                     "compression = 0 (BI_RGB)");

    // ---- Pixels: bottom row (y=1) first, BGR order ----
    // y=1 row at offset 54: white, black, grey, then 3 pad bytes
    check(f[54] == 255 && f[55] == 255 && f[56] == 255, "(0,1) white BGR");
    check(f[57] == 0   && f[58] == 0   && f[59] == 0,   "(1,1) black BGR");
    check(f[60] == 128 && f[61] == 128 && f[62] == 128, "(2,1) grey BGR");
    check(f[63] == 0 && f[64] == 0 && f[65] == 0,       "row 1 padding zeros");

    // y=0 row at offset 66: red, green, blue (BGR), then 3 pad bytes
    check(f[66] == 0   && f[67] == 0   && f[68] == 255, "(0,0) red as B0 G0 R255");
    check(f[69] == 0   && f[70] == 255 && f[71] == 0,   "(1,0) green as B0 G255 R0");
    check(f[72] == 255 && f[73] == 0   && f[74] == 0,   "(2,0) blue as B255 G0 R0");
    check(f[75] == 0 && f[76] == 0 && f[77] == 0,       "row 0 padding zeros");

    // ---- Out-of-bounds set_pixel is silently clipped ----
    {
        Bitmap b2(2, 2);
        b2.set_pixel(5, 5, 200, 200, 200);   // should not crash or corrupt
        b2.set_pixel(-1, 0, 200, 200, 200);
        save_bmp(path, b2);
        std::ifstream in2(path, std::ios::binary);
        bool ok = in2.good();
        in2.close();
        std::remove(path.c_str());
        check(ok, "out-of-bounds writes clipped, file still valid");
    }

    // ---- Non-positive dimensions throw ----
    {
        bool threw = false;
        try { Bitmap bad(0, 5); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "zero width throws");
    }

    std::cout << g_run - g_failed << " / " << g_run << " tests passed\n";
    return g_failed == 0 ? 0 : 1;
}

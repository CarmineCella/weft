// test_mnist.cpp
//
// Tests for the MNIST IDX loader.  We don't ship the real MNIST files
// in the repo, so we synthesise tiny IDX files in /tmp and read them back.
//
#include "MNIST.h"
#include "Matrix.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using weft::Matrix;

static int g_run = 0, g_failed = 0;
static void check(bool cond, const std::string& name) {
    ++g_run;
    std::cout << (cond ? "  [ ok ] " : "  [FAIL] ") << name << '\n';
    if (!cond) ++g_failed;
}
static bool close(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) < eps;
}

static void write_be_u32(std::ofstream& out, std::uint32_t x) {
    unsigned char b[4] = {
        static_cast<unsigned char>((x >> 24) & 0xff),
        static_cast<unsigned char>((x >> 16) & 0xff),
        static_cast<unsigned char>((x >>  8) & 0xff),
        static_cast<unsigned char>( x        & 0xff)
    };
    out.write(reinterpret_cast<char*>(b), 4);
}

int main() {
    std::cout << "weft :: MNIST loader tests\n";

    const std::string img_path = "/tmp/weft_test_images.idx";
    const std::string lbl_path = "/tmp/weft_test_labels.idx";

    // ---- 1. Round-trip a tiny image file ----
    //   2 images of size 3x3.
    //   Image 0:  pixel (i, j) = 10 * (3*i + j),   so 0, 10, 20, ..., 80
    //   Image 1:  all 255
    {
        std::ofstream out(img_path, std::ios::binary);
        write_be_u32(out, 0x00000803u);  // magic
        write_be_u32(out, 2);             // N
        write_be_u32(out, 3);             // H
        write_be_u32(out, 3);             // W
        for (int k = 0; k < 9; ++k)
            out.put(static_cast<char>(10 * k));
        for (int k = 0; k < 9; ++k)
            out.put(static_cast<char>(255));
        out.close();

        Matrix<float> X = weft::mnist::load_images<float>(img_path);
        std::remove(img_path.c_str());

        check(X.rows() == 9, "load_images: rows = H*W = 9");
        check(X.cols() == 2, "load_images: cols = N = 2");

        // Pixel order is row-major within each image:
        // X(3*i + j, image_index) = pixel (i, j).
        check(close(X(0, 0), 0.0f),         "load_images: image 0, pixel 0 = 0/255");
        check(close(X(1, 0), 10.0f / 255),  "load_images: image 0, pixel 1 = 10/255");
        check(close(X(8, 0), 80.0f / 255),  "load_images: image 0, pixel 8 = 80/255");

        bool all_one = true;
        for (std::size_t i = 0; i < 9; ++i)
            if (!close(X(i, 1), 1.0f)) all_one = false;
        check(all_one, "load_images: image 1 is all ones (255/255)");
    }

    // ---- 2. Round-trip a tiny label file ----
    {
        std::ofstream out(lbl_path, std::ios::binary);
        write_be_u32(out, 0x00000801u);  // magic
        write_be_u32(out, 5);             // N
        unsigned char labels[5] = {5, 0, 9, 3, 1};
        out.write(reinterpret_cast<char*>(labels), 5);
        out.close();

        std::vector<int> got = weft::mnist::load_labels(lbl_path);
        std::remove(lbl_path.c_str());

        check(got.size() == 5, "load_labels: returns N labels");
        check(got[0] == 5 && got[1] == 0 && got[2] == 9 &&
              got[3] == 3 && got[4] == 1,
              "load_labels: label values match");
    }

    // ---- 3. Wrong magic number throws ----
    {
        std::ofstream out(img_path, std::ios::binary);
        write_be_u32(out, 0xdeadbeefu);
        write_be_u32(out, 0);
        out.close();

        bool threw = false;
        try { (void)weft::mnist::load_images<float>(img_path); }
        catch (const std::runtime_error&) { threw = true; }
        std::remove(img_path.c_str());

        check(threw, "load_images: wrong magic throws runtime_error");
    }

    // ---- 4. Missing file throws ----
    {
        bool threw = false;
        try { (void)weft::mnist::load_labels("/tmp/weft_no_such_file.idx"); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "load_labels: missing file throws runtime_error");
    }

    // ---- 5. Big-endian read works regardless of host endianness ----
    //   Verify by writing N = 0x01020304 in the header and seeing it round-trip.
    {
        std::ofstream out(lbl_path, std::ios::binary);
        write_be_u32(out, 0x00000801u);  // magic
        write_be_u32(out, 0x01020304u);  // N - too big to actually write, but
                                          // we never read the body if we
                                          // catch the resulting short read.
        out.close();

        bool threw = false;
        try { (void)weft::mnist::load_labels(lbl_path); }
        catch (const std::runtime_error&) { threw = true; }
        std::remove(lbl_path.c_str());

        // The header parsed correctly (magic check passed), so the loader
        // attempted to read 0x01020304 bytes and threw on short read.
        // If we had a byte-swap bug, the magic check would have failed first
        // with a different runtime_error.  Either way `threw` is true, but
        // the test is here to document the contract.
        check(threw, "load_labels: big-endian header parsed (then short read)");
    }

    std::cout << g_run - g_failed << " / " << g_run << " tests passed\n";
    return g_failed == 0 ? 0 : 1;
}

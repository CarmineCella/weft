// tests/test_cifar10.cpp
//
// Tests for the CIFAR-10 loader.  Real CIFAR-10 lives at ~163 MB and
// can't be in the test data, so we synthesise tiny "fake CIFAR" files
// with the same byte layout (1 label byte + 1024 R + 1024 G + 1024 B
// per record) and verify the loader pulls the right values into the
// right NCHW positions.  If this passes, the loader will work on real
// data; if it fails, no amount of real data would help.

#include "../src/CIFAR10.h"

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>

using namespace weft;

static int passed = 0, total = 0;
static void check(bool cond, const char* name) {
    ++total;
    if (cond) ++passed;
    else      std::cout << "FAIL: " << name << "\n";
}

// Write a fake CIFAR batch file with the given records.
struct Rec {
    std::uint8_t label;
    // Per-channel value generators: cb(plane, h, w) returns the byte to
    // write for that plane.  Lets each test construct distinctive data.
};

static void write_fake_record(std::ofstream& f, std::uint8_t label,
                              std::uint8_t r_val, std::uint8_t g_val,
                              std::uint8_t b_val) {
    f.write(reinterpret_cast<const char*>(&label), 1);
    for (int k = 0; k < 1024; ++k) f.write(reinterpret_cast<const char*>(&r_val), 1);
    for (int k = 0; k < 1024; ++k) f.write(reinterpret_cast<const char*>(&g_val), 1);
    for (int k = 0; k < 1024; ++k) f.write(reinterpret_cast<const char*>(&b_val), 1);
}

int main() {
    std::cout << "--- testing CIFAR10 loader on synthetic data ---\n";

    const std::string path  = "/tmp/weft_fake_cifar.bin";
    const std::string path2 = "/tmp/weft_fake_cifar2.bin";

    // -----------------------------------------------------------------
    // Construct a 2-record file:
    //   record 0:  label=5, R=0, G=128, B=255
    //   record 1:  label=7, R varies linearly across the 1024 R bytes
    //              (so we can probe pixel ordering), G=0, B=0
    // -----------------------------------------------------------------
    {
        std::ofstream f(path, std::ios::binary);

        write_fake_record(f, /*label=*/5,
                          /*R=*/0, /*G=*/128, /*B=*/255);

        // Record 1: hand-built so R(h, w) is a known function of (h, w).
        std::uint8_t lbl = 7;
        f.write(reinterpret_cast<const char*>(&lbl), 1);
        for (int k = 0; k < 1024; ++k) {
            std::uint8_t r = static_cast<std::uint8_t>(k & 0xff);
            f.write(reinterpret_cast<const char*>(&r), 1);
        }
        for (int k = 0; k < 1024; ++k) {
            std::uint8_t g = 0;
            f.write(reinterpret_cast<const char*>(&g), 1);
        }
        for (int k = 0; k < 1024; ++k) {
            std::uint8_t b = 0;
            f.write(reinterpret_cast<const char*>(&b), 1);
        }
    }

    // -----------------------------------------------------------------
    std::cout << "  testing: loader returns the right shapes for one file\n";
    {
        auto data = load_cifar10<float>({path});
        check(data.images.N() == 2,
              "loader: 2 records produces N=2");
        check(data.images.C() == 3 && data.images.H() == 32 && data.images.W() == 32,
              "loader: (N, 3, 32, 32) tensor shape");
        check(data.labels.size() == 2,
              "loader: 2 records produces 2 labels");
    }

    std::cout << "  testing: label bytes are read correctly\n";
    {
        auto data = load_cifar10<float>({path});
        check(data.labels[0] == 5, "loader: first record's label is 5");
        check(data.labels[1] == 7, "loader: second record's label is 7");
    }

    std::cout << "  testing: pixel values are normalised to [0, 1]\n";
    {
        auto data = load_cifar10<float>({path});
        // Record 0: R=0 -> 0/255=0, G=128 -> 128/255, B=255 -> 1
        check(data.images(0, 0, 0, 0) == 0.0f,
              "record 0 channel R = 0/255");
        check(std::fabs(data.images(0, 1, 0, 0) - 128.0f/255.0f) < 1e-6f,
              "record 0 channel G = 128/255");
        check(data.images(0, 2, 0, 0) == 1.0f,
              "record 0 channel B = 255/255 = 1");
        // And the last pixel, to make sure the whole plane was read.
        check(data.images(0, 2, 31, 31) == 1.0f,
              "record 0 channel B last pixel = 1");
    }

    std::cout << "  testing: pixel ordering within a plane is row-major (h*32 + w)\n";
    {
        auto data = load_cifar10<float>({path});
        // Record 1 R values were written as byte index k for k=0..1023.
        // Our NCHW layout stores R(h, w) at flat offset h*32 + w within
        // the R plane, so the byte at index k should land at R(h, w)
        // with h = k/32, w = k%32.
        for (int k = 0; k < 1024; ++k) {
            const std::size_t h = static_cast<std::size_t>(k) / 32;
            const std::size_t w = static_cast<std::size_t>(k) % 32;
            const float expected = static_cast<float>(k & 0xff) / 255.0f;
            if (std::fabs(data.images(1, 0, h, w) - expected) > 1e-6f) {
                check(false, "row-major ordering R(h, w) = k where k = h*32 + w");
                break;
            }
            if (k == 1023) check(true, "row-major ordering R(h, w) = k where k = h*32 + w");
        }
    }

    std::cout << "  testing: G/B planes did not bleed into R\n";
    {
        auto data = load_cifar10<float>({path});
        // Record 1 had G=0 and B=0 throughout, so any non-zero in G or B
        // would mean the R plane bled across plane boundaries.
        bool g_clean = true, b_clean = true;
        for (std::size_t h = 0; h < 32 && (g_clean || b_clean); ++h)
            for (std::size_t w = 0; w < 32 && (g_clean || b_clean); ++w) {
                if (data.images(1, 1, h, w) != 0.0f) g_clean = false;
                if (data.images(1, 2, h, w) != 0.0f) b_clean = false;
            }
        check(g_clean, "channel separation: G plane unaffected by R contents");
        check(b_clean, "channel separation: B plane unaffected by R contents");
    }

    std::cout << "  testing: multiple files concatenate in order\n";
    {
        // Make a second file with one record so we can tell file order.
        std::ofstream f(path2, std::ios::binary);
        write_fake_record(f, /*label=*/9, /*R=*/10, /*G=*/20, /*B=*/30);
        f.close();

        auto data = load_cifar10<float>({path, path2});
        check(data.images.N() == 3,
              "loader: 2 + 1 records concatenate to N=3");
        check(data.labels[0] == 5 && data.labels[1] == 7 && data.labels[2] == 9,
              "loader: records appear in file-then-record order");
        // Spot-check the new record's pixel value.
        check(std::fabs(data.images(2, 0, 0, 0) - 10.0f/255.0f) < 1e-6f,
              "second file record 0 R = 10/255");
    }

    std::cout << "  testing: malformed file (wrong size) is rejected\n";
    {
        const std::string bad = "/tmp/weft_bad_cifar.bin";
        std::ofstream f(bad, std::ios::binary);
        const char garbage[] = "not a multiple of 3073";
        f.write(garbage, sizeof(garbage) - 1);
        f.close();
        bool threw = false;
        try { (void)load_cifar10<float>({bad}); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "loader: rejects files that are not a multiple of 3073 bytes");
        std::remove(bad.c_str());
    }

    std::cout << "  testing: class names match the canonical CIFAR-10 list\n";
    {
        check(std::string(cifar10_class_name(0)) == "airplane",
              "class name: 0 -> airplane");
        check(std::string(cifar10_class_name(3)) == "cat",
              "class name: 3 -> cat");
        check(std::string(cifar10_class_name(9)) == "truck",
              "class name: 9 -> truck");
    }

    std::remove(path.c_str());
    std::remove(path2.c_str());

    std::cout << passed << " / " << total << " tests passed.\n";
    return passed == total ? 0 : 1;
}

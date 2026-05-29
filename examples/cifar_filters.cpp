// cifar_filters.cpp
//
// Visualise the first convolutional layer's learned kernels as a single
// RGB grid image.  This is the payoff of the conv chapter: once trained
// on CIFAR-10, the 32 5x5 filters in Conv2D layer 0 stop looking random
// and start looking like oriented edge detectors, colour-opponent
// blobs, and low-frequency patterns -- the classical Krizhevsky-2012 /
// Zeiler-Fergus-2014 hierarchical-feature picture.
//
// Each filter has shape (in_C=3, K=5, K=5), so it's literally a tiny
// RGB image we can render directly.  In Conv2D the weights are kept as
// a Matrix of shape (out_C, in_C * K * K) = (32, 75), and each row is
// one filter, flattened in im2col row-order: row = c*K*K + kh*K + kw.
// So row 0 reshapes to filter 0's (3, 5, 5), etc.
//
// Each filter is normalised independently to fill [0, 255] so the
// pattern is visible regardless of its raw scale; then we scale each
// 5x5 RGB tile up by an integer factor for legibility and pack 32
// tiles into a 4x8 grid with a small black gutter between them.
//
// Usage:
//   cifar_filters <model_prefix> <out.bmp>
//   (model_prefix is the same one passed to cifar_conv; we read
//    prefix.conv and ignore prefix.dense -- only the first conv layer
//    is visualised.)

#include "ConvNetwork.h"
#include "Conv2D.h"
#include "ReLU4D.h"
#include "MaxPool2D.h"
#include "Bmp.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>

using namespace weft;

constexpr std::size_t IN_C  = 3;
constexpr std::size_t OUT_C = 32;
constexpr std::size_t K     = 5;
constexpr std::size_t GRID_ROWS  = 4;
constexpr std::size_t GRID_COLS  = 8;       // 4 x 8 = 32 filters
constexpr int         TILE_SCALE = 12;      // each 5x5 -> 60x60 in the BMP
constexpr int         GUTTER     = 2;       // pixels between tiles

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: cifar_filters <model_prefix> <out.bmp>\n";
        return 1;
    }
    const std::string prefix   = argv[1];
    const std::string out_path = argv[2];

    // -----------------------------------------------------------------
    // Re-build the same conv architecture as cifar_conv, then load the
    // saved weights into it.  ConvNetwork::load() requires the network
    // to be pre-built with the matching shape (same as Network::load).
    // -----------------------------------------------------------------
    ConvNetwork<float> conv;
    conv.add<Conv2D>  (/*in=*/3,  /*out=*/32, /*k=*/5, /*stride=*/1, /*pad=*/2);
    conv.add<ReLU4D>  ();
    conv.add<MaxPool2D>(2);
    conv.add<Conv2D>  (/*in=*/32, /*out=*/64, /*k=*/3, /*stride=*/1, /*pad=*/1);
    conv.add<ReLU4D>  ();
    conv.add<MaxPool2D>(2);

    try { conv.load(prefix + ".conv"); }
    catch (const std::exception& e) {
        std::cerr << "error loading " << prefix << ".conv: " << e.what() << "\n";
        return 1;
    }
    std::cout << "loaded " << prefix << ".conv\n";

    // The first conv layer is at index 0.  We need its weight matrix.
    // ConvNetwork doesn't expose layers directly (it owns unique_ptrs),
    // so we do a small dynamic_cast through a typed accessor.  Since we
    // KNOW the architecture (we just built it), this is safe.
    // We added Conv2D first via add<Conv2D>, which returns a reference,
    // but we threw that away above for brevity.  Rebuild + load doesn't
    // give us that handle back.  Solution: store the reference at add()
    // time so we can read its weights afterwards.
    //
    // (Refactored: do that in the actual flow below.)

    // -----------------------------------------------------------------
    // Do the architecture build again, this time keeping the Conv2D
    // reference returned by add() so we can inspect its weight matrix.
    // -----------------------------------------------------------------
    ConvNetwork<float> conv2;
    Conv2D<float>& first_conv =
        conv2.add<Conv2D>(/*in=*/3,  /*out=*/32, /*k=*/5, /*stride=*/1, /*pad=*/2);
    conv2.add<ReLU4D>  ();
    conv2.add<MaxPool2D>(2);
    conv2.add<Conv2D>  (/*in=*/32, /*out=*/64, /*k=*/3, /*stride=*/1, /*pad=*/1);
    conv2.add<ReLU4D>  ();
    conv2.add<MaxPool2D>(2);
    conv2.load(prefix + ".conv");

    const Matrix<float>& W = first_conv.W();   // (32, 3 * 5 * 5) = (32, 75)
    if (W.rows() != OUT_C || W.cols() != IN_C * K * K) {
        std::cerr << "unexpected first-layer shape: ("
                  << W.rows() << ", " << W.cols() << ")\n";
        return 1;
    }

    // -----------------------------------------------------------------
    // Compute the output canvas size.
    //
    //   Each filter tile is (K * TILE_SCALE) pixels on a side.
    //   Add a GUTTER of black pixels around every tile.
    // -----------------------------------------------------------------
    const int tile     = static_cast<int>(K) * TILE_SCALE;
    const int W_canvas = static_cast<int>(GRID_COLS) * tile
                       + static_cast<int>(GRID_COLS + 1) * GUTTER;
    const int H_canvas = static_cast<int>(GRID_ROWS) * tile
                       + static_cast<int>(GRID_ROWS + 1) * GUTTER;

    Bitmap bmp(W_canvas, H_canvas);   // default-black canvas

    // -----------------------------------------------------------------
    // For each of the 32 filters:
    //   1. Read its 75 weights (one row of W).
    //   2. Per-filter normalise to [0, 255]:  min->0, max->255.
    //      Per-filter rather than global so weak filters don't get
    //      washed out by strong ones.
    //   3. Splat the resulting 5x5 RGB tile, scaled up by TILE_SCALE.
    // -----------------------------------------------------------------
    for (std::size_t f = 0; f < OUT_C; ++f) {
        // Find per-filter min/max for normalisation.
        float mn =  1e30f, mx = -1e30f;
        for (std::size_t k = 0; k < IN_C * K * K; ++k) {
            const float v = W(f, k);
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        const float range = (mx > mn) ? (mx - mn) : 1.0f;

        // Grid position of this tile.
        const std::size_t gr = f / GRID_COLS;
        const std::size_t gc = f % GRID_COLS;
        const int x0 = static_cast<int>(gc + 1) * GUTTER
                     + static_cast<int>(gc)     * tile;
        const int y0 = static_cast<int>(gr + 1) * GUTTER
                     + static_cast<int>(gr)     * tile;

        // Splat each of the 5x5 kernel pixels.
        for (std::size_t kh = 0; kh < K; ++kh)
            for (std::size_t kw = 0; kw < K; ++kw) {
                // Channel order in W's row layout: c*K*K + kh*K + kw.
                const float r = W(f, 0 * K * K + kh * K + kw);
                const float g = W(f, 1 * K * K + kh * K + kw);
                const float b = W(f, 2 * K * K + kh * K + kw);
                auto to_u8 = [&](float v) {
                    return static_cast<std::uint8_t>(
                        std::min(255.0f,
                            std::max(0.0f, 255.0f * (v - mn) / range)));
                };
                const std::uint8_t R = to_u8(r);
                const std::uint8_t G = to_u8(g);
                const std::uint8_t B = to_u8(b);

                // Place the (kh, kw) source pixel as a TILE_SCALE block.
                for (int dy = 0; dy < TILE_SCALE; ++dy)
                    for (int dx = 0; dx < TILE_SCALE; ++dx) {
                        const int x = x0 + static_cast<int>(kw) * TILE_SCALE + dx;
                        const int y = y0 + static_cast<int>(kh) * TILE_SCALE + dy;
                        bmp.set_pixel(x, y, R, G, B);
                    }
            }
    }

    save_bmp(out_path, bmp);
    std::cout << "wrote " << out_path << "  ("
              << W_canvas << "x" << H_canvas << " px, "
              << OUT_C << " filters)\n";
    return 0;
}

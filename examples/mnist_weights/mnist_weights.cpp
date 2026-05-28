// mnist_weights / mnist_weights.cpp
//
// Visualising what a network learned by looking at its weights directly.
//
// A Dense layer's weight matrix has one ROW per output neuron and one
// COLUMN per input.  For an MNIST model the input is 784 pixels, so each
// row is 784 numbers -- which is just a 28x28 image.  Render that image
// and you see, literally, what the neuron is looking for.
//
// We use the simplest possible model: a linear classifier (784 -> 10,
// then softmax), no hidden layer.  Its ten weight rows are ten per-digit
// TEMPLATES.  Where a template is strongly positive, the model takes ink
// as evidence FOR that digit; where strongly negative, ink is evidence
// AGAINST it.  We draw this with a diverging colormap (red = positive,
// blue = negative, white ~ 0), which is exactly why Bmp.h writes RGB.
//
// (For a network with hidden layers you'd visualise the FIRST layer's
// rows the same way; they tend to look like stroke and edge detectors.
// The linear model is used here because its templates are the most
// directly interpretable.)
//
// Output:  mnist_weights.bmp  -- the ten templates in a 5x2 grid
//          (top row digits 0-4, bottom row 5-9)
//
// Usage:  mnist_weights [data_dir]   (default ../data/mnist)
//
#include "Adam.h"
#include "Bmp.h"
#include "CrossEntropy.h"
#include "Data.h"
#include "Dense.h"
#include "MNIST.h"
#include "Network.h"
#include "Softmax.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace weft;

// White-centred diverging colormap: t in [-1, 1] -> (r, g, b).
// t = 0 is white, t -> +1 reddens, t -> -1 blues.
static void diverging(float t, std::uint8_t& r, std::uint8_t& g, std::uint8_t& b) {
    t = std::max(-1.0f, std::min(1.0f, t));
    if (t >= 0.0f) {
        r = 255;
        g = static_cast<std::uint8_t>(255.0f * (1.0f - t));
        b = g;
    } else {
        b = 255;
        r = static_cast<std::uint8_t>(255.0f * (1.0f + t));
        g = r;
    }
}

// Draw row `row` of weight matrix W (one output neuron's 784 weights) as a
// 28x28 diverging-colormap tile, top-left at (x0, y0), scaled up.
static void blit_weights(Bitmap& bmp, const Matrix<float>& W, std::size_t row,
                         int x0, int y0, int scale, float maxabs) {
    for (int r = 0; r < 28; ++r)
        for (int c = 0; c < 28; ++c) {
            const float w = W(row, static_cast<std::size_t>(r * 28 + c));
            std::uint8_t R, G, B;
            diverging(maxabs > 0 ? w / maxabs : 0.0f, R, G, B);
            for (int dy = 0; dy < scale; ++dy)
                for (int dx = 0; dx < scale; ++dx)
                    bmp.set_pixel(x0 + c * scale + dx, y0 + r * scale + dy, R, G, B);
        }
}

int main(int argc, char** argv) {
    const std::string data_dir = (argc > 1) ? argv[1] : "../data/mnist";

    std::cout << "weft :: MNIST weight visualisation\n";
    std::cout << "data dir: " << data_dir << "\n\n";

    Matrix<float>    X_train, X_test;
    std::vector<int> y_train, y_test;
    try {
        X_train = mnist::load_images<float>(data_dir + "/train-images-idx3-ubyte");
        X_test  = mnist::load_images<float>(data_dir + "/t10k-images-idx3-ubyte");
        y_train = mnist::load_labels       (data_dir + "/train-labels-idx1-ubyte");
        y_test  = mnist::load_labels       (data_dir + "/t10k-labels-idx1-ubyte");
    } catch (const std::exception& e) {
        std::cerr << "error loading MNIST: " << e.what() << "\n";
        std::cerr << "did you run data/download_mnist.sh ?\n";
        return 1;
    }
    Matrix<float> Y_train = one_hot<float>(y_train, 10);
    Matrix<float> Y_test  = one_hot<float>(y_test, 10);
    std::cout << "loaded " << X_train.cols() << " train, "
              << X_test.cols() << " test images\n\n";

    // Linear classifier.  Keep a reference to the Dense layer so we can
    // read its weights after training (add<> returns a reference to the
    // layer it constructs; the Network holds it by unique_ptr, so the
    // reference stays valid as more layers are added).
    Network<float> net;
    Dense<float>&  fc = net.add<Dense>(784, 10);
    net.add<Softmax>();

    CrossEntropy<float> ce;
    Adam<float>         opt(1e-3f);
    const std::size_t   batch = 128;
    const int           epochs = 15;

    std::cout << "training linear classifier (784 -> 10 -> Softmax), "
              << epochs << " epochs...\n";

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        net.train();
        auto idx = shuffled_indices(X_train.cols(), epoch);
        for (std::size_t s = 0; s < X_train.cols(); s += batch) {
            const std::size_t e = std::min(s + batch, X_train.cols());
            std::vector<std::size_t> b(idx.begin() + s, idx.begin() + e);
            Matrix<float> Xb = X_train.selectColumns(b);
            Matrix<float> Yb = Y_train.selectColumns(b);
            Matrix<float> P  = net.forward(Xb);
            ce.forward(P, Yb);
            net.backward(ce.backward());
            net.update(opt);
        }
    }

    net.eval();
    std::cout << "test accuracy: "
              << std::fixed << std::setprecision(4)
              << accuracy<float>(net.forward(X_test), Y_test) << "\n";

    // ---- Render the ten weight templates ----
    const Matrix<float>& W = fc.W();        // (10 x 784)

    float maxabs = 0.0f;
    for (std::size_t i = 0; i < W.rows(); ++i)
        for (std::size_t j = 0; j < W.cols(); ++j)
            maxabs = std::max(maxabs, std::fabs(W(i, j)));

    const int scale = 6, cell = 28 * scale, gap = 6, cols = 5, rows = 2;
    Bitmap img(cols * cell + (cols - 1) * gap, rows * cell + (rows - 1) * gap);
    for (int d = 0; d < 10; ++d) {
        const int gx = (d % cols) * (cell + gap);
        const int gy = (d / cols) * (cell + gap);
        blit_weights(img, W, static_cast<std::size_t>(d), gx, gy, scale, maxabs);
    }
    save_bmp("mnist_weights.bmp", img);

    std::cout << "\nwrote mnist_weights.bmp (" << img.width() << "x" << img.height()
              << ", per-digit weight templates 0-4 top row, 5-9 bottom)\n"
              << "  red  = ink here is evidence FOR the digit\n"
              << "  blue = ink here is evidence AGAINST the digit\n";
    return 0;
}

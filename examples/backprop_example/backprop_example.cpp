// backprop_example.cpp
//
// A complete, hand-computed SGD step on a single neuron with two inputs,
// then the same step performed with weft::Dense for comparison. Both
// halves print every intermediate number; running the program is the
// fastest way to see that the matrix formulation in Dense::backward
// is doing exactly the same chain-rule arithmetic, only mass-produced.
//
// The accompanying note is notes/02b_one_weight.md.
//
// Where is the "squared error" loss in this code?
//   It appears in exactly one line:    dL_dz = z - t;
//   That is the derivative of  L = (1/2)(z - t)^2.  Everything downstream
//   (the chain rule, the SGD update) uses only that number, with no
//   knowledge of which loss produced it. Change the loss -> only that one
//   line changes.

#include "Dense.h"
#include "Matrix.h"

#include <iostream>
#include <iomanip>

int main() {
    using std::cout;
    cout << std::fixed << std::setprecision(4);

    // ------------------------------------------------------------------
    // PART 1 -- by hand, with plain scalars
    // ------------------------------------------------------------------
    cout << "=== Hand-computed SGD step on one neuron ===\n\n";

    const float x1 = 2.0f, x2 = -1.0f;
    float w1 = 0.5f, w2 = 1.0f, b = 0.1f;
    const float target = 1.0f;
    const float lr     = 0.1f;

    cout << "Inputs:    x1 = " << x1 << ",  x2 = " << x2 << '\n';
    cout << "Weights:   w1 = " << w1 << ",  w2 = " << w2 << ",  b = " << b << '\n';
    cout << "Target:    t  = " << target << '\n';
    cout << "lr (eta):  "      << lr     << "\n\n";

    // -- Forward pass (the projection) -------------------------------
    float z = w1 * x1 + w2 * x2 + b;
    cout << "Forward (projection):\n";
    cout << "  z = w1*x1 + w2*x2 + b = " << z << "\n\n";

    // -- Loss (squared error) ----------------------------------------
    float L = 0.5f * (z - target) * (z - target);
    cout << "Squared error loss:\n";
    cout << "  L = (1/2)(z - t)^2 = " << L << "\n\n";

    // -- Error signal at the output ----------------------------------
    //    dL/dz for L = (1/2)(z - t)^2 is just (z - t).
    //    This is the ONLY line where the choice of loss lives.
    float dL_dz = z - target;
    cout << "Error signal at the output:\n";
    cout << "  dL/dz = z - t = " << dL_dz << "\n\n";

    // -- Chain rule: dL/dp = dL/dz * dz/dp ---------------------------
    //    dz/dw1 = x1,  dz/dw2 = x2,  dz/db = 1
    float dL_dw1 = dL_dz * x1;
    float dL_dw2 = dL_dz * x2;
    float dL_db  = dL_dz * 1.0f;
    cout << "Gradients (chain rule  dL/dp = dL/dz * dz/dp):\n";
    cout << "  dL/dw1 = " << dL_dz << " * " << x1 << "  = " << dL_dw1 << '\n';
    cout << "  dL/dw2 = " << dL_dz << " * " << x2 << " = "  << dL_dw2 << '\n';
    cout << "  dL/db  = " << dL_dz << " * 1     = "         << dL_db  << "\n\n";

    // -- SGD step: against the gradient ------------------------------
    //    Notice: no mention of "squared error" here. SGD is generic.
    float w1_new = w1 - lr * dL_dw1;
    float w2_new = w2 - lr * dL_dw2;
    float b_new  = b  - lr * dL_db;
    cout << "SGD update (p <- p - eta * dL/dp):\n";
    cout << "  w1: " << w1 << " -> " << w1_new << '\n';
    cout << "  w2: " << w2 << " -> " << w2_new << '\n';
    cout << "  b : " << b  << " -> " << b_new  << "\n\n";

    w1 = w1_new; w2 = w2_new; b = b_new;

    // -- Verify: forward again with updated weights ------------------
    float z_new = w1 * x1 + w2 * x2 + b;
    float L_new = 0.5f * (z_new - target) * (z_new - target);
    cout << "Forward after update:\n";
    cout << "  z = " << z_new << "  (target " << target << ")\n";
    cout << "  L = " << L_new << "  (was " << L << ")\n\n";

    // ------------------------------------------------------------------
    // PART 2 -- the same step via weft::Dense
    // ------------------------------------------------------------------
    cout << "=== Same step with weft::Dense ===\n\n";

    using weft::Matrix;
    using weft::Dense;

    // Reset the layer to the same starting parameters as Part 1.
    Dense<float> layer(2, 1);
    layer.W() = Matrix<float>{{0.5f, 1.0f}};
    layer.b() = Matrix<float>{{0.1f}};

    Matrix<float> X{{2.0f}, {-1.0f}};   // (2 in x 1 example)
    Matrix<float> t{{1.0f}};            // (1 out x 1 example)

    Matrix<float> Z  = layer.forward(X);
    Matrix<float> dZ = Z - t;           // dL/dZ for squared error
    layer.backward(dZ);

    cout << "Forward:           z = "  << Z(0, 0)            << '\n';
    cout << "Error signal:  dL/dz = "  << dZ(0, 0)           << '\n';
    cout << "dW from backward(): [ " << layer.dW()(0, 0)
                                  << ", " << layer.dW()(0, 1) << " ]\n";
    cout << "db from backward(): [ " << layer.db()(0, 0) << " ]\n\n";

    layer.update(lr);
    cout << "After update():\n";
    cout << "  W = [ " << layer.W()(0, 0) << ", " << layer.W()(0, 1) << " ]\n";
    cout << "  b = [ " << layer.b()(0, 0) << " ]\n\n";

    cout << "These numbers match the hand calculation exactly.\n";
    cout << "Dense::backward is doing the same chain-rule arithmetic,\n";
    cout << "but mass-produced over every weight (and every example in\n";
    cout << "the batch) via a single matrix multiply.\n";

    return 0;
}

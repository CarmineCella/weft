#pragma once
//
// Network.h  --  an ordered list of layers, run forward and backward as a unit.
//
//   forward(X)        :  feed X through the layers in order, return final output.
//   backward(dY)      :  feed dY back through the layers in reverse, return dL/dX.
//                        Side effect: every layer's parameter gradients are set.
//   update(opt)       :  hand each layer's parameters to the optimiser.
//   train() / eval()  :  propagate training/inference mode to every layer.
//                        Layers default to training mode; only call eval() when
//                        running on a validation/test set.
//
//   add<L>(args...)   :  construct a new layer of type L<T> with the given
//                        constructor args and append it.  Returns a reference
//                        to the just-added layer, so callers can configure it
//                        (e.g. set weights for a test).
//
// The loss function lives OUTSIDE the network: a Network produces predictions,
// a Loss compares them to targets and starts the backward pass with its dL/dS.
//
//   Example:
//       Network<float> net;
//       net.add<Dense>(4, 16);
//       net.add<ReLU>();
//       net.add<Dense>(16, 3);
//       net.add<Softmax>();
//
//       CrossEntropy<float> ce;
//       SGD<float>          opt(0.01f);
//
//       Matrix<float> S  = net.forward(X);
//       float        L   = ce.forward(S, Targets);
//       Matrix<float> dS = ce.backward();
//       net.backward(dS);
//       net.update(opt);
//
#include "Layer.h"
#include "Matrix.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace weft {

template <typename T = float>
class Network {
public:
    // Construct a new layer of type L<T> in place and append it.
    // Returns a reference to the just-added layer (concrete type, not Layer<T>&)
    // so the caller can read or set its parameters directly.
    //
    //   auto& d = net.add<Dense>(4, 16);
    //   d.W() = ...;
    //
    template <template <typename> class L, typename... Args>
    L<T>& add(Args&&... args) {
        auto ptr = std::make_unique<L<T>>(std::forward<Args>(args)...);
        L<T>* raw = ptr.get();
        layers_.push_back(std::move(ptr));
        return *raw;
    }

    // Forward: every layer in order.
    Matrix<T> forward(const Matrix<T>& X) {
        Matrix<T> A = X;
        for (auto& layer : layers_)
            A = layer->forward(A);
        return A;
    }

    // Backward: every layer in reverse, threading the gradient.
    // Returns dL/dX (gradient w.r.t. the network's input).
    Matrix<T> backward(const Matrix<T>& dY) {
        Matrix<T> g = dY;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
            g = (*it)->backward(g);
        return g;
    }

    // Apply one optimiser step to every layer.  Layers without parameters
    // (activations) just no-op.
    void update(Optimizer<T>& opt) {
        for (auto& layer : layers_)
            layer->update(opt);
    }

    // Switch every layer to training mode (the default).
    void train() {
        for (auto& layer : layers_)
            layer->set_training(true);
    }

    // Switch every layer to inference mode.  Dropout becomes identity,
    // and any future stateful normaliser stops updating its running stats.
    void eval() {
        for (auto& layer : layers_)
            layer->set_training(false);
    }

    std::size_t size() const { return layers_.size(); }

    // Multi-line description of the network, one layer per line, indented
    // four spaces, no trailing newline.  Built from each layer's describe(),
    // so the printed architecture is generated from the real layers and
    // can't drift out of sync the way a hand-written string can.
    std::string summary() const {
        std::string out;
        for (std::size_t i = 0; i < layers_.size(); ++i) {
            out += "    " + layers_[i]->describe();
            if (i + 1 < layers_.size()) out += "\n";
        }
        return out;
    }

    // Save / load trainable parameters to a binary file.  The file records
    // a magic tag, a version, and the layer count; each layer then writes
    // its own parameters (Dense writes weights+bias; activations write
    // nothing).  load() requires an already-constructed network with the
    // SAME architecture -- it fills in the weights, it does not build the
    // graph.  Layer-count and per-matrix shape checks catch a mismatch.
    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Network::save: cannot open " + path);
        out.write("WNET", 4);
        const std::uint32_t version = 1;
        const std::uint32_t n = static_cast<std::uint32_t>(layers_.size());
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (const auto& layer : layers_) layer->save_params(out);
        if (!out) throw std::runtime_error("Network::save: write failed: " + path);
    }

    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Network::load: cannot open " + path);
        char magic[4];
        in.read(magic, 4);
        if (!in || std::string(magic, 4) != "WNET")
            throw std::runtime_error("Network::load: bad magic in " + path);
        std::uint32_t version = 0, n = 0;
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (n != layers_.size())
            throw std::runtime_error(
                "Network::load: layer count mismatch (build the same "
                "architecture before loading)");
        for (auto& layer : layers_) layer->load_params(in);
        if (!in) throw std::runtime_error("Network::load: read failed: " + path);
    }

private:
    std::vector<std::unique_ptr<Layer<T>>> layers_;
};

} // namespace weft

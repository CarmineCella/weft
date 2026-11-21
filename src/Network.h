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
//
//       SoftmaxCrossEntropy<float> ce;  // consumes the Dense logits
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
#include "Checkpoint.h"

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
    // graph. Version 2 also checks exact layer configuration, scalar width,
    // and caller-supplied preprocessing metadata. Legacy version 1 has only
    // layer-count/shape checks and cannot satisfy a metadata requirement.
    void save(const std::string& path, const std::string& metadata = "") const {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Network::save: cannot open " + path);
        detail::write_checkpoint_header<T>(out, "WNET",
            static_cast<std::uint32_t>(layers_.size()), configuration(metadata));
        for (const auto& layer : layers_) layer->save_params(out);
        out.flush();
        if (!out) throw std::runtime_error("Network::save: write failed: " + path);
    }

    void load(const std::string& path, const std::string& metadata = "") {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Network::load: cannot open " + path);
        detail::read_checkpoint_header<T>(in, "WNET",
            static_cast<std::uint32_t>(layers_.size()), configuration(metadata),
            metadata.empty());
        for (auto& layer : layers_) layer->load_params(in);
        if (!in) throw std::runtime_error("Network::load: read failed: " + path);
    }

private:
    std::string configuration(const std::string& metadata) const {
        std::string config;
        for (const auto& layer : layers_) {
            const std::string part = layer->configuration();
            config += std::to_string(part.size()) + ":" + part;
        }
        return config + "\n" + metadata;
    }
    std::vector<std::unique_ptr<Layer<T>>> layers_;
};

} // namespace weft

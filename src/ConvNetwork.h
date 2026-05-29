#pragma once
//
// ConvNetwork.h  --  container for ConvLayer instances.
//
// Parallel to Network<T>, with exactly the same interface so the user
// can build a conv stack the same way they build a dense one:
//
//      ConvNetwork<float> conv;
//      conv.add<Conv2D>(/*in=*/1, /*out=*/8, /*k=*/3, /*stride=*/1, /*pad=*/1);
//      conv.add<ReLU4D>();
//      conv.add<MaxPool2D>(2);
//      // ... etc.
//
// The classic conv-then-dense classifier (AlexNet shape) is then
// composed at the example level:
//
//      Tensor4D<float> feat = conv.forward(X);
//      Matrix<float>   flat = flatten(feat);
//      Matrix<float>   prob = dense.forward(flat);
//
// add() returns a reference to the just-added layer so the caller can
// keep a typed handle for parameter inspection or per-layer logging.
// std::unique_ptr storage means moving the vector around doesn't
// invalidate that reference (the layers stay heap-allocated).
//
#include "ConvLayer.h"
#include "Optimizer.h"

#include <memory>
#include <vector>
#include <sstream>
#include <string>
#include <fstream>
#include <cstdint>
#include <stdexcept>

namespace weft {

template <typename T = float>
class ConvNetwork {
public:
    // ---------------------------------------------------------------
    // Building the network.  L must derive from ConvLayer<T>.
    // ---------------------------------------------------------------
    template <template <typename> class L, typename... Args>
    L<T>& add(Args&&... args) {
        auto p = std::make_unique<L<T>>(std::forward<Args>(args)...);
        L<T>& ref = *p;
        layers_.push_back(std::move(p));
        return ref;
    }

    std::size_t size() const { return layers_.size(); }

    // ---------------------------------------------------------------
    // Forward / backward / update.  Identical pattern to Network<T>.
    // ---------------------------------------------------------------
    Tensor4D<T> forward(const Tensor4D<T>& X) {
        Tensor4D<T> A = X;
        for (auto& l : layers_) A = l->forward(A);
        return A;
    }

    Tensor4D<T> backward(const Tensor4D<T>& dY) {
        Tensor4D<T> g = dY;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
            g = (*it)->backward(g);
        return g;
    }

    void update(Optimizer<T>& opt) {
        for (auto& l : layers_) l->update(opt);
    }

    void train() { for (auto& l : layers_) l->set_training(true);  }
    void eval () { for (auto& l : layers_) l->set_training(false); }

    // ---------------------------------------------------------------
    // Human-readable architecture summary, one layer per line.
    // ---------------------------------------------------------------
    std::string summary() const {
        std::ostringstream ss;
        ss << "ConvNetwork(" << layers_.size() << " layers)\n";
        for (std::size_t i = 0; i < layers_.size(); ++i)
            ss << "  [" << i << "] " << layers_[i]->describe() << '\n';
        return ss.str();
    }

    // ---------------------------------------------------------------
    // Persistence.  Same on-disk format as Network<T> but with a
    // distinct magic so a conv file isn't loaded as a dense file by
    // mistake.  Header:  4-byte magic "WCNV" + u32 version + u32
    // layer_count, then each layer's save_params payload back-to-back.
    // Loading requires the network to already be built with the same
    // architecture (same layer count, same shapes).
    // ---------------------------------------------------------------
    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("ConvNetwork::save: cannot open " + path);
        const char magic[4] = {'W','C','N','V'};
        const std::uint32_t version = 1;
        const std::uint32_t count   = static_cast<std::uint32_t>(layers_.size());
        f.write(magic, 4);
        f.write(reinterpret_cast<const char*>(&version), sizeof(version));
        f.write(reinterpret_cast<const char*>(&count),   sizeof(count));
        for (const auto& l : layers_) l->save_params(f);
    }

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("ConvNetwork::load: cannot open " + path);
        char magic[4];
        std::uint32_t version, count;
        f.read(magic, 4);
        f.read(reinterpret_cast<char*>(&version), sizeof(version));
        f.read(reinterpret_cast<char*>(&count),   sizeof(count));
        if (magic[0]!='W'||magic[1]!='C'||magic[2]!='N'||magic[3]!='V')
            throw std::runtime_error("ConvNetwork::load: bad magic");
        if (count != layers_.size())
            throw std::runtime_error("ConvNetwork::load: layer count mismatch");
        for (auto& l : layers_) l->load_params(f);
    }

private:
    std::vector<std::unique_ptr<ConvLayer<T>>> layers_;
};

} // namespace weft

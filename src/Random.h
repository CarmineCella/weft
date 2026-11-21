#pragma once

#include <cstdint>
#include <random>

namespace weft {
namespace detail {
inline std::mt19937& seed_engine() {
    static thread_local std::mt19937 engine(std::random_device{}());
    return engine;
}
} // namespace detail

// Call before constructing a model. Each layer gets an independent stream;
// reseeding reproduces initialization and dropout on the calling thread.
// Standard distributions need not be bit-identical across C++ implementations.
inline void set_seed(std::uint32_t seed) { detail::seed_engine().seed(seed); }
inline std::uint32_t random_seed() { return detail::seed_engine()(); }

} // namespace weft

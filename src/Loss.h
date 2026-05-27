#pragma once
//
// Loss.h  --  abstract base class for loss functions.
//
//   forward(predicted, target)  ->  scalar loss (mean across the batch)
//                                   caches whatever backward needs.
//
//   backward()                  ->  dL/d(predicted)
//                                   shape matches `predicted`.
//
// CONVENTION: the mean-loss factor 1/N (where N is the batch size, i.e.
// the number of columns of `predicted`) is baked into the gradient
// returned by backward().  Layer backward formulas therefore stay clean
// -- they never know or care about batch size.
//
#include "Matrix.h"

namespace weft {

template <typename T = float>
class Loss {
public:
    virtual ~Loss() = default;
    virtual T          forward(const Matrix<T>& predicted,
                               const Matrix<T>& target) = 0;
    virtual Matrix<T>  backward() = 0;
};

} // namespace weft

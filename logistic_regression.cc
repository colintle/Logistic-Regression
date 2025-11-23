#include "logistic_regression.h"

#include <cmath>
#include <stdexcept>
#include <vector>

LogisticRegression::LogisticRegression(size_t input_dim, size_t num_classes,
                                       double learning_rate, int iterations)
    : learning_rate_(learning_rate), iterations_(iterations),
      weights_(matrix_library::Tensor<double>({input_dim, 1})),
      bias_(matrix_library::Tensor<double>({1, num_classes})),
      num_classes_(num_classes) {}

void LogisticRegression::validate_inputs(
    const matrix_library::Tensor<double> &X,
    const matrix_library::Tensor<double> &y) const {
  const auto &xshape = X.shape();
  const auto &yshape = y.shape();

  if (num_classes_ == 0)
    throw std::logic_error(
        "num_classes_ must be set via constructor before validation");

  // X must be 2D: (N, M)
  if (xshape.size() != 2)
    throw std::invalid_argument("X must be a 2D tensor with shape (N, M)");

  size_t N = xshape[0];
  size_t M = xshape[1];

  if (N == 0)
    throw std::invalid_argument("N must be greater than 0");

  // y must be 1D: (N)
  if (yshape.size() != 1)
    throw std::invalid_argument("y must be a 1D vector of length N");
  if (yshape[0] != N)
    throw std::invalid_argument("y length must match X's number of rows (N)");

  // M must match model input size (weights_.shape()[0])
  const auto &wshape = weights_.shape();
  if (wshape.size() != 2)
    throw std::logic_error("weights_ must be 2D with shape (M, C)");
  if (M != wshape[0])
    throw std::invalid_argument(
        "X has incorrect number of features (M) for this model");
}

void LogisticRegression::fit(const matrix_library::Tensor<double> &X,
                             const matrix_library::Tensor<double> &y) {
  // Validate shapes and sizes before training
  validate_inputs(X, y);
  // TODO: implement training
}

matrix_library::Tensor<double> LogisticRegression::one_hot_encode(
    const matrix_library::Tensor<double> &y) const {
  const auto &shape = y.shape();
  size_t N = shape[0];

  constexpr double eps = 1e-9;
  size_t C = num_classes_;
  matrix_library::Tensor<double> out({N, C});
  out.fill(0.0);
  for (size_t i = 0; i < N; ++i) {
    double v = y[{i}];
    double r = std::round(v);
    if (std::fabs(v - r) > eps)
      throw std::invalid_argument("y contains non-integer class labels");
    if (r < 0.0)
      throw std::invalid_argument("y contains negative class labels");
    size_t lbl = static_cast<size_t>(r);
    if (lbl >= C)
      throw std::invalid_argument(
          "Class label out of range given number of classes");
    out[{i, lbl}] = 1.0;
  }
  return out;
}

const matrix_library::Tensor<double> &LogisticRegression::get_weights() const {
  return weights_;
}
const matrix_library::Tensor<double> &LogisticRegression::get_bias() const {
  return bias_;
}

#include "logistic_regression.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "operations/add.h"
#include "operations/multiply.h"
#include "operations/scale.h"
#include "operations/subtract.h"
#include "operations/transpose.h"

LogisticRegression::LogisticRegression(size_t input_dim, size_t num_classes,
                                       double learning_rate, int iterations)
    : learning_rate_(learning_rate), iterations_(iterations),
      weights_(matrix_library::Tensor<double>({input_dim, num_classes})),
      bias_(matrix_library::Tensor<double>({1, num_classes})),
      num_classes_(num_classes) {}

void LogisticRegression::validate_inputs(
    const matrix_library::Tensor<double> &X,
    const matrix_library::Tensor<double> &y) const {
  const auto &x_shape = X.shape();
  const auto &y_shape = y.shape();

  if (num_classes_ == 0)
    throw std::logic_error(
        "num_classes_ must be set via constructor before validation");

  if (x_shape.size() != 2)
    throw std::invalid_argument("X must be a 2D tensor with shape (N, M)");

  size_t N = x_shape[0];
  size_t M = x_shape[1];

  if (N == 0)
    throw std::invalid_argument("N must be greater than 0");

  if (y_shape.size() != 1)
    throw std::invalid_argument("y must be a 1D vector of length N");
  if (y_shape[0] != N)
    throw std::invalid_argument("y length must match X's number of rows (N)");

  const auto &w_shape = weights_.shape();
  if (w_shape.size() != 2)
    throw std::logic_error("weights_ must be 2D with shape (M, C)");
  if (M != w_shape[0])
    throw std::invalid_argument(
        "X has incorrect number of features (M) for this model");
}

void LogisticRegression::fit(const matrix_library::Tensor<double> &X,
                             const matrix_library::Tensor<double> &y) {
  validate_inputs(X, y);
  const size_t N = X.shape()[0];
  const size_t M = X.shape()[1];
  const size_t C = num_classes_;

  // Precompute a row vector of ones for bias gradient: shape (1, N)
  matrix_library::Tensor<double> ones_row({1, N});
  for (size_t i = 0; i < N; ++i)
    ones_row[{0, i}] = 1.0;

  // One-hot encode targets once (shape: N x C)
  matrix_library::Tensor<double> Y = one_hot_encode(y);

  for (int iter = 0; iter < iterations_; ++iter) {
    // logits = X @ W + b  -> (N, C)
    auto logits = matrix_library::operations::multiply(X, weights_);
    matrix_library::Tensor<double> bias_tiled({N, C});
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < C; ++j)
        bias_tiled[{i, j}] = bias_[{0, j}];
    logits = matrix_library::operations::add(logits, bias_tiled);

    // probs = softmax(logits) row-wise
    matrix_library::Tensor<double> probs({N, C});
    for (size_t i = 0; i < N; ++i) {
      std::vector<double> row(C);
      for (size_t j = 0; j < C; ++j)
        row[j] = logits[{i, j}];
      softmax(row);
      for (size_t j = 0; j < C; ++j)
        probs[{i, j}] = row[j];
    }

    // diff = probs - Y  -> (N, C)
    auto diff = matrix_library::operations::subtract(probs, Y);

    // dW = (1/N) * (X^T @ diff) -> (M, C)
    auto XT = matrix_library::operations::transpose(X);
    auto dW = matrix_library::operations::multiply(XT, diff);
    dW = matrix_library::operations::scale(dW, 1.0 / static_cast<double>(N));

    // dB = (1/N) * (ones_row @ diff) -> (1, C)
    auto dB = matrix_library::operations::multiply(ones_row, diff);
    dB = matrix_library::operations::scale(dB, 1.0 / static_cast<double>(N));

    // Update parameters
    weights_ = matrix_library::operations::subtract(
        weights_, matrix_library::operations::scale(dW, learning_rate_));
    bias_ = matrix_library::operations::subtract(
        bias_, matrix_library::operations::scale(dB, learning_rate_));
  }
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

matrix_library::Tensor<double>
LogisticRegression::predict(const matrix_library::Tensor<double> &X) const {
  const auto &x_shape = X.shape();
  if (x_shape.size() != 2)
    throw std::invalid_argument("X must be a 2D tensor with shape (K, M)");

  const size_t K = x_shape[0];
  const size_t M = x_shape[1];

  const auto &w_shape = weights_.shape();
  if (w_shape.size() != 2)
    throw std::logic_error("weights_ must be 2D with shape (M, C)");
  if (M != w_shape[0])
    throw std::invalid_argument(
        "Input feature dimension does not match model weights.");

  const size_t C = w_shape[1];
  if (K == 0) {
    return matrix_library::Tensor<double>({0, C});
  }

  // logits = X @ W + b -> (K, C)
  auto logits = matrix_library::operations::multiply(X, weights_);
  matrix_library::Tensor<double> bias_tiled({K, C});
  for (size_t i = 0; i < K; ++i)
    for (size_t j = 0; j < C; ++j)
      bias_tiled[{i, j}] = bias_[{0, j}];
  logits = matrix_library::operations::add(logits, bias_tiled);

  // probs = softmax(logits) row-wise
  matrix_library::Tensor<double> probs({K, C});
  for (size_t i = 0; i < K; ++i) {
    std::vector<double> row(C);
    for (size_t j = 0; j < C; ++j)
      row[j] = logits[{i, j}];
    softmax(row);
    for (size_t j = 0; j < C; ++j)
      probs[{i, j}] = row[j];
  }
  return probs;
}

void LogisticRegression::softmax(std::vector<double> &row) {
  if (row.empty())
    return;
  double max_val = row[0];
  for (double v : row)
    if (v > max_val)
      max_val = v;

  double sum = 0.0;
  for (double &v : row) {
    v = std::exp(v - max_val);
    sum += v;
  }
  if (sum == 0.0) {
    double u = 1.0 / static_cast<double>(row.size());
    for (double &v : row)
      v = u;
  } else {
    for (double &v : row)
      v /= sum;
  }
}

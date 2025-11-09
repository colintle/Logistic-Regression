#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "tensor.h"

// N is the number of samples
// M is the number of features
// K is the number of samples to predict
// Output dimension is 1 for binary classification (0 or 1)
class LogisticRegression {
public:
  LogisticRegression(size_t input_size, double learning_rate, int iterations);

  LogisticRegression(const LogisticRegression &) = delete;
  LogisticRegression &operator=(const LogisticRegression &) = delete;

  // X shape: (N, M)
  // y shape: (N, 1)
  void fit(const matrix_library::Tensor<double> &X,
           const matrix_library::Tensor<double> &y);
  // X shape: (K, M)
  // Output shape: (K, 1)
  double predict(const matrix_library::Tensor<double> &X) const;

  const matrix_library::Tensor<double> &get_weights() const;
  const matrix_library::Tensor<double> &get_bias() const;

private:
  double sigmoid(double z) const;

  double learning_rate_;
  int iterations_;
  // Weights shape: (M, 1)
  matrix_library::Tensor<double> weights_;
  // Bias shape: (1, 1)
  matrix_library::Tensor<double> bias_;
};

#endif // LOGISTIC_REGRESSION_H
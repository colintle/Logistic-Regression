#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "tensor.h"

// N is the number of samples
// M is the number of features
// K is the number of samples to predict
// C is the number of classes
class LogisticRegression {
public:
  LogisticRegression(size_t input_size, double learning_rate, int iterations);

  LogisticRegression(const LogisticRegression &) = delete;
  LogisticRegression &operator=(const LogisticRegression &) = delete;

  // X shape: (N, M)
  // y shape: (N, 1) with class labels (0 to C-1)
  void fit(const matrix_library::Tensor<double> &X,
           const matrix_library::Tensor<double> &y);
  // X shape: (K, M)
  // Output shape: (K, C) with probabilities for each class
  matrix_library::Tensor<double>
  predict(const matrix_library::Tensor<double> &X) const;

  const matrix_library::Tensor<double> &get_weights() const;
  const matrix_library::Tensor<double> &get_bias() const;

private:
  // Row shape: (1, C)
  // Modifies the input row in-place
  static void softmax(std::vector<double> &row);
  double learning_rate_;
  int iterations_;
  // Weights shape: (M, C)
  matrix_library::Tensor<double> weights_;
  // Bias shape: (1, C)
  matrix_library::Tensor<double> bias_;
  // Number of classes
  size_t num_classes_ = 0;
};

#endif // LOGISTIC_REGRESSION_H
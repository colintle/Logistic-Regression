#include <gtest/gtest.h>

#include "logistic_regression.h"
#include "operations/add.h"
#include "operations/multiply.h"

using matrix_library::Tensor;

namespace {

static Tensor<double> logits_from_params(const Tensor<double> &X,
                                         const Tensor<double> &W,
                                         const Tensor<double> &b) {
  const size_t N = X.shape()[0];
  const size_t C = W.shape()[1];
  auto logits = matrix_library::operations::multiply(X, W);
  Tensor<double> bias_tiled({N, C});
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < C; ++j)
      bias_tiled[{i, j}] = b[{0, j}];
  return matrix_library::operations::add(logits, bias_tiled);
}

static size_t argmax_row(const Tensor<double> &mat, size_t row) {
  const size_t C = mat.shape()[1];
  size_t best = 0;
  double bestv = mat[{row, 0}];
  for (size_t j = 1; j < C; ++j) {
    double v = mat[{row, j}];
    if (v > bestv) {
      bestv = v;
      best = j;
    }
  }
  return best;
}

} // namespace

TEST(LogisticRegressionTest, FitsLinearlySeparableBinary) {
  const size_t N = 8;
  const size_t M = 2;
  const size_t C = 2;

  Tensor<double> X({N, M});
  X[{0, 0}] = 0.0;
  X[{0, 1}] = 0.0;
  X[{1, 0}] = 0.0;
  X[{1, 1}] = 1.0;
  X[{2, 0}] = 1.0;
  X[{2, 1}] = 0.0;
  X[{3, 0}] = 0.5;
  X[{3, 1}] = 0.2;
  X[{4, 0}] = 3.0;
  X[{4, 1}] = 3.0;
  X[{5, 0}] = 4.0;
  X[{5, 1}] = 4.0;
  X[{6, 0}] = 3.0;
  X[{6, 1}] = 4.0;
  X[{7, 0}] = 4.0;
  X[{7, 1}] = 3.0;

  Tensor<double> y({N});
  y[{0}] = 0;
  y[{1}] = 0;
  y[{2}] = 0;
  y[{3}] = 0;
  y[{4}] = 1;
  y[{5}] = 1;
  y[{6}] = 1;
  y[{7}] = 1;

  LogisticRegression lr(M, C, /*lr*/ 0.1, /*iters*/ 1500);
  lr.fit(X, y);

  auto logits = logits_from_params(X, lr.get_weights(), lr.get_bias());

  for (size_t i = 0; i < N; ++i) {
    size_t pred = argmax_row(logits, i);
    EXPECT_EQ(pred, static_cast<size_t>(y[{i}])) << "i=" << i;
  }
}

TEST(LogisticRegressionTest, FitsLinearlySeparableMulticlass) {
  const size_t N = 6;
  const size_t M = 2;
  const size_t C = 3;

  Tensor<double> X({N, M});
  X[{0, 0}] = 0.0;
  X[{0, 1}] = 0.0;
  X[{1, 0}] = 0.2;
  X[{1, 1}] = 0.1;
  X[{2, 0}] = 4.0;
  X[{2, 1}] = 0.0;
  X[{3, 0}] = 5.0;
  X[{3, 1}] = 0.2;
  X[{4, 0}] = 0.0;
  X[{4, 1}] = 4.0;
  X[{5, 0}] = 0.1;
  X[{5, 1}] = 5.0;

  Tensor<double> y({N});
  y[{0}] = 0;
  y[{1}] = 0;
  y[{2}] = 1;
  y[{3}] = 1;
  y[{4}] = 2;
  y[{5}] = 2;

  LogisticRegression lr(M, C, /*lr*/ 0.1, /*iters*/ 2000);
  lr.fit(X, y);

  auto logits = logits_from_params(X, lr.get_weights(), lr.get_bias());
  for (size_t i = 0; i < N; ++i) {
    size_t pred = argmax_row(logits, i);
    EXPECT_EQ(pred, static_cast<size_t>(y[{i}])) << "i=" << i;
  }
}

TEST(LogisticRegressionTest, ValidateInputsErrors) {
  const size_t N = 4;
  const size_t M = 2;
  const size_t C = 2;
  LogisticRegression lr(M, C, 0.1, 10);

  Tensor<double> X({N, M});
  Tensor<double> y2d({N, 1});
  for (size_t i = 0; i < N; ++i) {
    y2d[{i, 0}] = 0;
  }
  EXPECT_THROW(lr.fit(X, y2d), std::invalid_argument);

  Tensor<double> X0({0, M});
  Tensor<double> y0({0});
  EXPECT_THROW(lr.fit(X0, y0), std::invalid_argument);

  Tensor<double> y_bad({N - 1});
  for (size_t i = 0; i + 1 < N; ++i)
    y_bad[{i}] = 0;
  EXPECT_THROW(lr.fit(X, y_bad), std::invalid_argument);
}

TEST(LogisticRegressionTest, PredictOnTrainedBinary) {
  const size_t N = 8;
  const size_t M = 2;
  const size_t C = 2;

  Tensor<double> X({N, M});
  X[{0, 0}] = 0.0;
  X[{0, 1}] = 0.0;
  X[{1, 0}] = 0.0;
  X[{1, 1}] = 1.0;
  X[{2, 0}] = 1.0;
  X[{2, 1}] = 0.0;
  X[{3, 0}] = 0.5;
  X[{3, 1}] = 0.2;
  X[{4, 0}] = 3.0;
  X[{4, 1}] = 3.0;
  X[{5, 0}] = 4.0;
  X[{5, 1}] = 4.0;
  X[{6, 0}] = 3.0;
  X[{6, 1}] = 4.0;
  X[{7, 0}] = 4.0;
  X[{7, 1}] = 3.0;

  Tensor<double> y({N});
  y[{0}] = 0;
  y[{1}] = 0;
  y[{2}] = 0;
  y[{3}] = 0;
  y[{4}] = 1;
  y[{5}] = 1;
  y[{6}] = 1;
  y[{7}] = 1;

  LogisticRegression lr(M, C, /*lr*/ 0.1, /*iters*/ 1500);
  lr.fit(X, y);

  auto probs = lr.predict(X);
  ASSERT_EQ(probs.shape().size(), 2u);
  ASSERT_EQ(probs.shape()[0], N);
  ASSERT_EQ(probs.shape()[1], C);

  for (size_t i = 0; i < N; ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < C; ++j) {
      double p = probs[{i, j}];
      EXPECT_GE(p, 0.0);
      EXPECT_LE(p, 1.0);
      sum += p;
    }
    EXPECT_NEAR(sum, 1.0, 1e-6);

    size_t pred = argmax_row(probs, i);
    EXPECT_EQ(pred, static_cast<size_t>(y[{i}])) << "i=" << i;
  }
}

TEST(LogisticRegressionTest, PredictOnTrainedMulticlass) {
  const size_t N = 6;
  const size_t M = 2;
  const size_t C = 3;

  Tensor<double> X({N, M});
  X[{0, 0}] = 0.0;
  X[{0, 1}] = 0.0;
  X[{1, 0}] = 0.2;
  X[{1, 1}] = 0.1;
  X[{2, 0}] = 4.0;
  X[{2, 1}] = 0.0;
  X[{3, 0}] = 5.0;
  X[{3, 1}] = 0.2;
  X[{4, 0}] = 0.0;
  X[{4, 1}] = 4.0;
  X[{5, 0}] = 0.1;
  X[{5, 1}] = 5.0;

  Tensor<double> y({N});
  y[{0}] = 0;
  y[{1}] = 0;
  y[{2}] = 1;
  y[{3}] = 1;
  y[{4}] = 2;
  y[{5}] = 2;

  LogisticRegression lr(M, C, /*lr*/ 0.1, /*iters*/ 2000);
  lr.fit(X, y);

  auto probs = lr.predict(X);
  ASSERT_EQ(probs.shape().size(), 2u);
  ASSERT_EQ(probs.shape()[0], N);
  ASSERT_EQ(probs.shape()[1], C);

  for (size_t i = 0; i < N; ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < C; ++j) {
      double p = probs[{i, j}];
      EXPECT_GE(p, 0.0);
      EXPECT_LE(p, 1.0);
      sum += p;
    }
    EXPECT_NEAR(sum, 1.0, 1e-6);

    size_t pred = argmax_row(probs, i);
    EXPECT_EQ(pred, static_cast<size_t>(y[{i}])) << "i=" << i;
  }
}

TEST(LogisticRegressionTest, PredictEmptyInput) {
  const size_t M = 2;
  const size_t C = 3;
  LogisticRegression lr(M, C, 0.1, 1);
  Tensor<double> X0({0, M});
  auto probs = lr.predict(X0);
  ASSERT_EQ(probs.shape().size(), 2u);
  EXPECT_EQ(probs.shape()[0], 0u);
  EXPECT_EQ(probs.shape()[1], C);
}

TEST(LogisticRegressionTest, PredictFeatureMismatchThrows) {
  const size_t M = 2;
  const size_t C = 2;
  LogisticRegression lr(M, C, 0.1, 1);
  Tensor<double> Xbad({1, M + 1});
  EXPECT_THROW(lr.predict(Xbad), std::invalid_argument);
}
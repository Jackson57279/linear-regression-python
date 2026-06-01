"""Unit tests for LinearRegression and matrix utilities.

Run with:  python -m unittest test_linear_regression.py -v
"""
import math
import unittest

from linear_regression import (
    LinearRegression, add_bias, matmul, matvec, mean_squared_error,
    r2_score, standardize, train_test_split, transpose,
)


class TestMatrixUtils(unittest.TestCase):
    def test_transpose(self):
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        self.assertEqual(transpose(A), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    def test_matmul_identity(self):
        I = [[1.0, 0.0], [0.0, 1.0]]
        A = [[1.5, -2.0], [3.25, 0.5]]
        self.assertEqual(matmul(A, I), A)

    def test_matvec(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        v = [5.0, 6.0]
        self.assertEqual(matvec(A, v), [17.0, 39.0])

    def test_add_bias(self):
        X = [[1.0, 2.0], [3.0, 4.0]]
        self.assertEqual(add_bias(X), [[1.0, 1.0, 2.0], [1.0, 3.0, 4.0]])


class TestLinearRegression(unittest.TestCase):
    def test_perfect_linear_fit(self):
        # y = 2 + 3*x1 + 4*x2
        X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
        y = [2.0, 5.0, 6.0, 9.0, 12.0]
        m = LinearRegression().fit(X, y)
        yhat = m.predict(X)
        for a, b in zip(y, yhat):
            self.assertAlmostEqual(a, b, places=8)
        # weights are [bias, w1, w2] == [2, 3, 4]
        self.assertAlmostEqual(m.weights[0], 2.0, places=8)
        self.assertAlmostEqual(m.weights[1], 3.0, places=8)
        self.assertAlmostEqual(m.weights[2], 4.0, places=8)

    def test_univariate_line(self):
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [3.0, 5.0, 7.0, 9.0, 11.0]  # y = 1 + 2x
        m = LinearRegression().fit(X, y)
        self.assertAlmostEqual(m.weights[0], 1.0, places=8)
        self.assertAlmostEqual(m.weights[1], 2.0, places=8)

    def test_ridge_shrinks(self):
        rng = [0.1 * i for i in range(20)]
        X = [[r] for r in rng]
        y = [3.0 * r + 0.05 * ((-1) ** i) for i, r in enumerate(rng)]
        ols = LinearRegression(alpha=0.0).fit(X, y)
        ridge = LinearRegression(alpha=10.0).fit(X, y)
        self.assertLess(abs(ridge.weights[1]), abs(ols.weights[1]))

    def test_no_intercept(self):
        X = [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 2.0]]
        y = [2.0, 4.0, 8.0, 10.0]  # y = 2*x1 + 4*x2
        m = LinearRegression(fit_intercept=False).fit(X, y)
        self.assertAlmostEqual(m.weights[0], 2.0, places=8)
        self.assertAlmostEqual(m.weights[1], 4.0, places=8)

    def test_predict_before_fit_raises(self):
        m = LinearRegression()
        with self.assertRaises(RuntimeError):
            m.predict([[0.0]])

    def test_metrics(self):
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.1, 1.9, 3.05, 3.95]
        # mean of (0.1, -0.1, 0.05, -0.05)^2 = 0.00625
        self.assertAlmostEqual(mean_squared_error(y_true, y_pred), 0.00625, places=8)
        # predictions match truth almost exactly -> R^2 close to 1
        self.assertGreater(r2_score(y_true, y_pred), 0.99)

    def test_r2_perfect_is_one(self):
        y = [0.5, -1.0, 2.0, 3.5]
        self.assertAlmostEqual(r2_score(y, y), 1.0, places=12)

    def test_standardize_train_test_match(self):
        X_tr = [[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]]
        X_te = [[2.0, 20.0], [4.0, 40.0]]
        Xt, Xe, mu, sd = standardize(X_tr, X_te)
        # training means approximately 0
        for j in range(2):
            col_mean = sum(Xt[i][j] for i in range(3)) / 3
            self.assertAlmostEqual(col_mean, 0.0, places=8)
        # test set uses the training-set statistics, not its own
        self.assertAlmostEqual(Xe[0][0], (2.0 - mu[0]) / sd[0], places=12)

    def test_train_test_split_sizes(self):
        X = [[float(i)] for i in range(100)]
        y = [float(i) for i in range(100)]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, seed=1)
        self.assertEqual(len(Xtr), 75)
        self.assertEqual(len(Xte), 25)
        # every element appears exactly once across the two splits
        seen = sorted([v[0] for v in Xtr] + [v[0] for v in Xte])
        self.assertEqual(seen, list(range(100)))

    def test_noisy_data_r2_in_range(self):
        # Sanity: with moderate noise the R^2 on training data is in (0, 1)
        import random as _r
        _r.seed(0)
        X = [[float(i), float(i) ** 0.5] for i in range(50)]
        y = [3.0 * x[0] - 1.0 * x[1] + _r.gauss(0, 0.5) for x in X]
        m = LinearRegression().fit(X, y)
        yhat = m.predict(X)
        r2 = r2_score(y, yhat)
        self.assertGreater(r2, 0.9)
        self.assertLessEqual(r2, 1.0)


if __name__ == "__main__":
    unittest.main()

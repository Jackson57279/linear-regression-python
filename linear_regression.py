"""
LinearRegression: from-scratch OLS linear regression with optional L2 (Ridge)
regularisation, trained via the closed-form Normal Equation.

Pure Python standard library only (no numpy / scikit-learn).

The model predicts a continuous target y as a linear combination of features:
    y_hat = X @ w + b

The closed-form solution minimises the regularised MSE cost:
    J(w, b) = (1/n) * sum((y - X@w - b)^2) + alpha * (w . w)

By absorbing the bias into the weight vector (adding a column of 1s to X),
the gradient w.r.t. w becomes zero at:
    w* = (X^T X + alpha * I')^{-1} X^T y
where I' has a 0 in the (0,0) position so the bias is not regularised.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

Vector = List[float]
Matrix = List[List[float]]


def _shape(X: Matrix) -> Tuple[int, int]:
    return len(X), len(X[0]) if X else 0


def _zeros(rows: int, cols: int) -> Matrix:
    return [[0.0] * cols for _ in range(rows)]


def transpose(X: Matrix) -> Matrix:
    n, d = _shape(X)
    return [[X[i][j] for i in range(n)] for j in range(d)]


def matmul(A: Matrix, B: Matrix) -> Matrix:
    n, k = _shape(A)
    k2, m = _shape(B)
    assert k == k2, f"shape mismatch: {n}x{k} * {k2}x{m}"
    out = _zeros(n, m)
    for i in range(n):
        Ai = A[i]
        out_i = out[i]
        for kk in range(k):
            a = Ai[kk]
            Bk = B[kk]
            for j in range(m):
                out_i[j] += a * Bk[j]
    return out


def matvec(A: Matrix, v: Vector) -> Vector:
    n, k = _shape(A)
    assert len(v) == k
    out = [0.0] * n
    for i in range(n):
        s = 0.0
        Ai = A[i]
        for j in range(k):
            s += Ai[j] * v[j]
        out[i] = s
    return out



def add_bias(X: Matrix) -> Matrix:
    """Prepend a column of 1s to X for the bias term."""
    return [[1.0] + X[i] for i in range(len(X))]



def _mat_inverse(A_in: Matrix) -> Matrix:
    """Gauss-Jordan inversion with partial pivoting."""
    n, m = _shape(A_in)
    assert n == m, "inverse requires a square matrix"
    A = [row[:] for row in A_in]
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        pivot = i
        for r in range(i + 1, n):
            if abs(A[r][i]) > abs(A[pivot][i]):
                pivot = r
        if abs(A[pivot][i]) < 1e-15:
            raise ValueError("matrix is singular (or near-singular)")
        if pivot != i:
            A[i], A[pivot] = A[pivot], A[i]
            I[i], I[pivot] = I[pivot], I[i]
        piv = A[i][i]
        for j in range(n):
            A[i][j] /= piv
            I[i][j] /= piv
        for r in range(n):
            if r == i:
                continue
            factor = A[r][i]
            if factor == 0.0:
                continue
            for j in range(n):
                A[r][j] -= factor * A[i][j]
                I[r][j] -= factor * I[i][j]
    return I


@dataclass
class LinearRegression:
    alpha: float = 0.0
    fit_intercept: bool = True
    weights: Vector = field(default_factory=list)

    def fit(self, X: Matrix, y: Vector) -> "LinearRegression":
        n = len(X)
        assert len(y) == n, "X and y must have the same number of rows"
        Z = add_bias(X) if self.fit_intercept else [row[:] for row in X]
        ZT = transpose(Z)
        d = len(Z[0])
        G = matmul(ZT, Z)
        for j in range(d):
            if self.fit_intercept and j == 0:
                continue
            G[j][j] += self.alpha
        rhs = matvec(ZT, y)
        G_inv = _mat_inverse(G)
        self.weights = matvec(G_inv, rhs)
        return self

    def predict(self, X: Matrix) -> Vector:
        if not self.weights:
            raise RuntimeError("model has not been fitted yet")
        Z = add_bias(X) if self.fit_intercept else [row[:] for row in X]
        return matvec(Z, self.weights)


def mean_squared_error(y_true, y_pred) -> float:
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n


def r2_score(y_true, y_pred) -> float:
    assert len(y_true) == len(y_pred)
    mean = sum(y_true) / len(y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((a - mean) ** 2 for a in y_true)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def train_test_split(X, y, test_size: float = 0.2, seed=42):
    n = len(X)
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    cut = int(n * (1 - test_size))
    tr, te = idx[:cut], idx[cut:]
    return ([X[i] for i in tr], [X[i] for i in te], [y[i] for i in tr], [y[i] for i in te])


def standardize(X_train, X_test):
    n, d = _shape(X_train)
    means = [0.0] * d
    for i in range(n):
        for j in range(d):
            means[j] += X_train[i][j]
    means = [m / n for m in means]
    stds = [0.0] * d
    for i in range(n):
        for j in range(d):
            diff = X_train[i][j] - means[j]
            stds[j] += diff * diff
    stds = [math.sqrt(s / n) if s > 0 else 1.0 for s in stds]
    Xt = [[(X_train[i][j] - means[j]) / stds[j] for j in range(d)] for i in range(n)]
    Xe = [[(X_test[i][j] - means[j]) / stds[j] for j in range(d)] for i in range(len(X_test))]
    return Xt, Xe, means, stds

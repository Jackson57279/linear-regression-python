# Linear Regression (Python)

A from-scratch **ordinary least squares** linear regression with optional
**L2 (Ridge) regularisation**, trained via the **closed-form Normal Equation**.
Pure Python standard library only — **no numpy, no scipy, no scikit-learn**.

The matrix utilities (`transpose`, `matmul`, `matvec`, `add_bias`,
`Gauss-Jordan inverse`) are implemented in `linear_regression.py` so that
every line of the algorithm is auditable.

## The math

The model predicts a continuous target `y` as a linear combination of
features:

```
y_hat = X @ w + b
```

We absorb the bias into the weight vector by prepending a column of 1s
to `X`, then minimise the regularised mean-squared error:

```
J(w) = (1/n) * sum((y - X@w)^2) + alpha * (w . w)
```

Setting `dJ/dw = 0` yields the **Normal Equation**:

```
w* = (X^T X + alpha * I')^{-1} X^T y
```

where `I'` is the identity matrix with a 0 in the `(0,0)` position so
the bias is **not** regularised.

## Project layout

| File | Purpose |
|---|---|
| `linear_regression.py` | `LinearRegression` class, matrix utilities, metrics |
| `data_loader.py` | Downloads and parses the California Housing dataset |
| `demo.py` | Trains OLS / Ridge on California Housing and reports MSE / R² |
| `test_linear_regression.py` | 14 unit tests, run with `unittest` |

## Quick start

```bash
cd linear-regression/python
python3 demo.py
```

Expected output (on first run the dataset is auto-downloaded):

```
Loaded 20433 rows, 8 features.
alpha= 0.0: test MSE = 0.4631, test R^2 = 0.6534
alpha= 1.0: test MSE = 0.4630, test R^2 = 0.6534
alpha=10.0: test MSE = 0.4630, test R^2 = 0.6534
```

The first run downloads `data/california_housing.csv` (~400 KB) from
the Aurélien Géron *Hands-On ML* mirror; subsequent runs are offline.

## Run the tests

```bash
python3 -m unittest test_linear_regression -v
```

All 14 tests cover the matrix utilities, a perfect-fit check, ridge
shrinking, no-intercept mode, error states, metrics, and a noisy-data
R² sanity check.

## Use it like scikit-learn

```python
from linear_regression import LinearRegression, standardize

X_train_s, X_test_s, _, _ = standardize(X_train, X_test)
model = LinearRegression(alpha=1.0).fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)
```

## Notes & limitations

- The matrix inverse uses **Gauss-Jordan with partial pivoting**, which is
  numerically fine for the small feature counts here but is not the
  production tool of choice. For larger problems use a Cholesky or QR
  decomposition (drop-in replacement in `LinearRegression.fit`).
- The dataset is cached in `data/`. Delete that directory to force a
  re-download.

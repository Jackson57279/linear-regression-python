"""Train and evaluate LinearRegression on the California Housing dataset."""
from __future__ import annotations

from data_loader import load
from linear_regression import (
    LinearRegression, mean_squared_error, r2_score,
    standardize, train_test_split,
)


def main() -> None:
    X, y = load()
    print(f"Loaded {len(X)} rows, {len(X[0])} features.")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, seed=42)
    X_tr_s, X_te_s, _, _ = standardize(X_tr, X_te)

    for alpha in (0.0, 1.0, 10.0):
        model = LinearRegression(alpha=alpha).fit(X_tr_s, y_tr)
        yhat = model.predict(X_te_s)
        mse = mean_squared_error(y_te, yhat)
        r2 = r2_score(y_te, yhat)
        print(f"alpha={alpha:>4}: test MSE = {mse:.4f}, test R^2 = {r2:.4f}")


if __name__ == "__main__":
    main()

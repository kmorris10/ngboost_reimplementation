"""
run_bonus_experiment.py

[BONUS] Proposed improvement to the baseline NGBoost reimplementation.

MOTIVATION
-----------
NGBoost's Normal(mu, sigma) output distribution places probability mass on
the entire real line, including negative values. The Concrete Compressive
Strength target, however, is a strictly positive physical quantity
(observed range 2.33-82.6 MPa, sample skewness = 0.42, i.e. moderately
right-skewed) -- exactly the situation the original NGBoost paper itself
flags as a case where "the predicted distributions need to have ... enough
degrees of freedom" and where a Normal assumption may be a poor match to
the outcome's support and shape.

We therefore propose swapping the assumed output distribution for a
Log-Normal distribution: y ~ LogNormal(mu, sigma) <=> log(y) ~ N(mu, sigma).
This requires no new natural-gradient derivation -- LogNormal boosting can
be implemented exactly by reusing our existing NGBoostNormal machinery
(ngboost_scratch.py) unchanged, but fitting it on z = log(y) instead of y,
and converting the evaluation back to the original y-scale via the
standard change-of-variables (Jacobian) correction:
    p_Y(y) = p_Z(log y) / y   =>   NLL_Y(y) = NLL_Z(log y) + log(y)
Point predictions are converted back using the Log-Normal mean,
E[Y] = exp(mu + sigma^2 / 2), so that RMSE is comparable on the original
MPa scale.

We expect this change to help most on datasets, like this one, where the
target is positive and skewed, because it (a) guarantees predictions stay
positive and (b) lets the model's implied noise scale naturally with the
predicted magnitude (multiplicative rather than additive noise).

This script reruns the exact same 20-repeat, train/val-select-M/refit/test
protocol as run_experiment.py (same random seeds, so the comparison is
paired), but fits the Log-Normal variant, and reports NLL/RMSE against the
already-computed baseline Normal NGBoost results in results_raw.csv.
"""

import numpy as np
import pandas as pd

from ngboost_scratch import NGBoostNormal

DATA_PATH = "Concrete_Data.csv"
N_REPEATS = 20
TEST_FRAC = 0.10
VAL_FRAC_OF_REMAINING = 0.20
MAX_STAGES = 800
LEARNING_RATE = 0.01
MAX_DEPTH = 3


def lognormal_nll_on_original_scale(mu_log, sigma_log, y):
    """NLL_Y(y) = NLL_Z(log y) + log(y), per the Jacobian correction above."""
    logy = np.log(y)
    nll_log_space = (np.log(sigma_log) + 0.5 * np.log(2 * np.pi)
                      + 0.5 * ((logy - mu_log) / sigma_log) ** 2)
    return (nll_log_space + logy).mean()


def lognormal_mean(mu_log, sigma_log):
    """E[Y] for Y ~ LogNormal(mu, sigma)."""
    return np.exp(mu_log + 0.5 * sigma_log ** 2)


def rmse(pred, y):
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def run_one_repeat_lognormal(X, y, rng):
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(TEST_FRAC * n)
    test_idx, rest_idx = idx[:n_test], idx[n_test:]

    n_val = int(VAL_FRAC_OF_REMAINING * len(rest_idx))
    val_idx, train_idx = rest_idx[:n_val], rest_idx[n_val:]

    X_train, y_train = X[train_idx], np.log(y[train_idx])
    X_val, y_val = X[val_idx], np.log(y[val_idx])
    X_rest, y_rest = X[rest_idx], np.log(y[rest_idx])
    X_test, y_test_original = X[test_idx], y[test_idx]

    search_model = NGBoostNormal(
        n_estimators=MAX_STAGES, learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH, random_state=int(rng.randint(0, 1_000_000)),
    )
    search_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    best_m = max(int(np.argmin(search_model.val_nll_)), 5)

    final_model = NGBoostNormal(
        n_estimators=best_m, learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH, random_state=int(rng.randint(0, 1_000_000)),
    )
    final_model.fit(X_rest, y_rest)

    mu_log_test, sigma_log_test = final_model.predict_dist(X_test)
    nll = lognormal_nll_on_original_scale(mu_log_test, sigma_log_test, y_test_original)
    point_pred = lognormal_mean(mu_log_test, sigma_log_test)
    return nll, rmse(point_pred, y_test_original), best_m


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(float)

    baseline = pd.read_csv("results_raw.csv")

    lognormal_nlls, lognormal_rmses = [], []
    for rep in range(N_REPEATS):
        rng = np.random.RandomState(rep)  # identical seed/splits to the baseline run
        nll, rm, m = run_one_repeat_lognormal(X, y, rng)
        lognormal_nlls.append(nll)
        lognormal_rmses.append(rm)
        print(f"[repeat {rep+1:2d}/{N_REPEATS}] Log-Normal NGBoost: "
              f"NLL(y-scale)={nll:.3f} RMSE={rm:.3f} (M={m})")

    ln_nll = np.array(lognormal_nlls)
    ln_rmse = np.array(lognormal_rmses)
    base_nll = baseline["ours_nll"].values
    base_rmse = baseline["ours_rmse"].values

    print("\n" + "=" * 78)
    print("BONUS COMPARISON -- Normal vs. Log-Normal output distribution")
    print("(paired, identical 20 train/val/test splits)")
    print("=" * 78)
    print(f"{'Variant':<28}{'Test NLL':<20}{'Test RMSE':<20}")
    print(f"{'Baseline (Normal)':<28}{base_nll.mean():.3f} +/- {base_nll.std():.3f}      "
          f"{base_rmse.mean():.3f} +/- {base_rmse.std():.3f}")
    print(f"{'Improvement (Log-Normal)':<28}{ln_nll.mean():.3f} +/- {ln_nll.std():.3f}      "
          f"{ln_rmse.mean():.3f} +/- {ln_rmse.std():.3f}")

    nll_wins = int((ln_nll < base_nll).sum())
    rmse_wins = int((ln_rmse < base_rmse).sum())
    print(f"\nLog-Normal beats Normal on NLL in {nll_wins}/{N_REPEATS} paired repeats")
    print(f"Log-Normal beats Normal on RMSE in {rmse_wins}/{N_REPEATS} paired repeats")

    pd.DataFrame({
        "repeat": range(1, N_REPEATS + 1),
        "baseline_nll": base_nll, "lognormal_nll": ln_nll,
        "baseline_rmse": base_rmse, "lognormal_rmse": ln_rmse,
    }).to_csv("results_bonus.csv", index=False)
    print("\nSaved: results_bonus.csv")


if __name__ == "__main__":
    main()

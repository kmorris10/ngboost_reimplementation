"""
run_experiment.py

Reproduces the core experiment of Duan et al. (2020), "NGBoost: Natural
Gradient Boosting for Probabilistic Prediction" (ICML 2020) on the UCI
"Concrete Compressive Strength" dataset (N=1030), which is one of the ten
benchmark datasets in the paper's Table 1 (probabilistic regression, NLL)
and Table 3 (point estimation, RMSE).

WHAT THIS SCRIPT DOES
----------------------
For 20 independent repeats (matching the paper's protocol, Section 4):
  1. Hold out a random 10% of the data as a test set.
  2. From the remaining 90%, hold out a further 20% as a validation set.
  3. Fit NGBoost on the 72%-train split, tracking validation NLL at every
     boosting stage, and pick the number of stages M* that minimises it.
  4. Refit NGBoost on the full 90% for M* stages.
  5. Evaluate on the held-out 10% test set: report NLL and RMSE.
This is done twice per repeat:
  (a) using our from-scratch implementation (ngboost_scratch.NGBoostNormal),
      which is the "reimplementation" required by the assignment; and
  (b) using the original authors' public `ngboost` PyPI package, run under
      an identical protocol, as an independent reference to sanity-check
      our reimplementation and to discuss any discrepancies.

DATASET SOURCE
---------------
UCI Machine Learning Repository, "Concrete Compressive Strength" dataset
(I-Cheng Yeh, 1998), N=1030, 8 features -> 1 continuous target. Retrieved
via a GitHub-hosted mirror of the UCI file (direct UCI archive access was
not reachable from this environment):
https://raw.githubusercontent.com/HansenHan/ML_UCI_Concrete_Compressive_Strength/main/Concrete_Data.csv

CITATION FOR THE REIMPLEMENTED ALGORITHM
------------------------------------------
Duan, T., Avati, A., Ding, D. Y., Thai, K. K., Basu, S., Ng, A., & Schuler,
A. (2020). NGBoost: Natural Gradient Boosting for Probabilistic Prediction.
Proceedings of the 37th ICML, PMLR 119. arXiv:1910.03225.
Official code (reference only, not copied): github.com/stanfordmlgroup/ngboost
"""

import time
import numpy as np
import pandas as pd

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore

from ngboost_scratch import NGBoostNormal

# ----------------------------------------------------------------------
# Configuration (matches the paper's protocol for the Concrete dataset)
# ----------------------------------------------------------------------
DATA_PATH = "Concrete_Data.csv"
N_REPEATS = 20          # paper repeats each UCI experiment 20 times
TEST_FRAC = 0.10
VAL_FRAC_OF_REMAINING = 0.20
MAX_STAGES = 800        # upper bound on boosting stages searched for M*
LEARNING_RATE = 0.01    # paper default for all UCI datasets except Year MSD
MAX_DEPTH = 3           # paper default tree depth

# Published numbers from Duan et al. (2020), Table 1 (NLL) and Table 3 (RMSE)
# for the Concrete dataset, quoted here for side-by-side comparison.
PAPER_NLL_MEAN, PAPER_NLL_STD = 3.04, 0.17
PAPER_RMSE_MEAN, PAPER_RMSE_STD = 5.06, 0.61


def neg_log_likelihood(mu, sigma, y):
    return (np.log(sigma) + 0.5 * np.log(2 * np.pi) + 0.5 * ((y - mu) / sigma) ** 2).mean()


def rmse(mu, y):
    return float(np.sqrt(np.mean((mu - y) ** 2)))


def run_one_repeat_ours(X, y, rng):
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(TEST_FRAC * n)
    test_idx, rest_idx = idx[:n_test], idx[n_test:]

    n_val = int(VAL_FRAC_OF_REMAINING * len(rest_idx))
    val_idx, train_idx = rest_idx[:n_val], rest_idx[n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_rest, y_rest = X[rest_idx], y[rest_idx]     # full 90% for the refit
    X_test, y_test = X[test_idx], y[test_idx]

    # Stage 1: search for the best number of boosting stages M* on 72%/18%.
    search_model = NGBoostNormal(
        n_estimators=MAX_STAGES, learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH, random_state=int(rng.randint(0, 1_000_000)),
    )
    search_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    best_m = int(np.argmin(search_model.val_nll_))  # index 0 = before any stage
    best_m = max(best_m, 5)  # avoid a degenerate 0-stage model

    # Stage 2: refit from scratch on the full 90% for M* stages.
    final_model = NGBoostNormal(
        n_estimators=best_m, learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH, random_state=int(rng.randint(0, 1_000_000)),
    )
    final_model.fit(X_rest, y_rest)

    mu_test, sigma_test = final_model.predict_dist(X_test)
    return neg_log_likelihood(mu_test, sigma_test, y_test), rmse(mu_test, y_test), best_m


def run_one_repeat_official(X, y, rng):
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(TEST_FRAC * n)
    test_idx, rest_idx = idx[:n_test], idx[n_test:]

    n_val = int(VAL_FRAC_OF_REMAINING * len(rest_idx))
    val_idx, train_idx = rest_idx[:n_val], rest_idx[n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_rest, y_rest = X[rest_idx], y[rest_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    search_model = NGBRegressor(
        Dist=Normal, Score=LogScore, n_estimators=MAX_STAGES,
        learning_rate=LEARNING_RATE, minibatch_frac=1.0, verbose=False,
        random_state=int(rng.randint(0, 1_000_000)),
    )
    search_model.fit(X_train, y_train, X_val=X_val, Y_val=y_val)
    best_m = max(int(search_model.best_val_loss_itr) + 1, 5)

    final_model = NGBRegressor(
        Dist=Normal, Score=LogScore, n_estimators=best_m,
        learning_rate=LEARNING_RATE, minibatch_frac=1.0, verbose=False,
        random_state=int(rng.randint(0, 1_000_000)),
    )
    final_model.fit(X_rest, y_rest)

    pred = final_model.pred_dist(X_test)
    mu_test, sigma_test = pred.params["loc"], pred.params["scale"]
    return neg_log_likelihood(mu_test, sigma_test, y_test), rmse(mu_test, y_test), best_m


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(float)
    print(f"Loaded Concrete Compressive Strength dataset: N={len(y)}, D={X.shape[1]} features")

    results = {"ours": {"nll": [], "rmse": [], "M": []},
               "official": {"nll": [], "rmse": [], "M": []}}

    t_start = time.time()
    for rep in range(N_REPEATS):
        rng = np.random.RandomState(rep)  # one seed drives both models' identical splits
        nll_o, rmse_o, m_o = run_one_repeat_ours(X, y, np.random.RandomState(rep))
        nll_f, rmse_f, m_f = run_one_repeat_official(X, y, np.random.RandomState(rep))
        results["ours"]["nll"].append(nll_o)
        results["ours"]["rmse"].append(rmse_o)
        results["ours"]["M"].append(m_o)
        results["official"]["nll"].append(nll_f)
        results["official"]["rmse"].append(rmse_f)
        results["official"]["M"].append(m_f)
        print(f"[repeat {rep+1:2d}/{N_REPEATS}] "
              f"ours: NLL={nll_o:.3f} RMSE={rmse_o:.3f} (M={m_o})  |  "
              f"official: NLL={nll_f:.3f} RMSE={rmse_f:.3f} (M={m_f})")

    elapsed = time.time() - t_start
    print(f"\nTotal run time: {elapsed:.1f} sec for {N_REPEATS} repeats x 2 models\n")

    def summarize(vals):
        arr = np.array(vals)
        return arr.mean(), arr.std()

    ours_nll_mean, ours_nll_std = summarize(results["ours"]["nll"])
    ours_rmse_mean, ours_rmse_std = summarize(results["ours"]["rmse"])
    off_nll_mean, off_nll_std = summarize(results["official"]["nll"])
    off_rmse_mean, off_rmse_std = summarize(results["official"]["rmse"])

    print("=" * 78)
    print("RESULTS -- Concrete Compressive Strength (N=1030), 20 repeats")
    print("=" * 78)
    print(f"{'Method':<28}{'Test NLL':<20}{'Test RMSE':<20}")
    print(f"{'Duan et al. (2020) paper':<28}{PAPER_NLL_MEAN:.2f} +/- {PAPER_NLL_STD:.2f}      "
          f"{PAPER_RMSE_MEAN:.2f} +/- {PAPER_RMSE_STD:.2f}")
    print(f"{'Ours (from scratch)':<28}{ours_nll_mean:.2f} +/- {ours_nll_std:.2f}      "
          f"{ours_rmse_mean:.2f} +/- {ours_rmse_std:.2f}")
    print(f"{'Official ngboost package':<28}{off_nll_mean:.2f} +/- {off_nll_std:.2f}      "
          f"{off_rmse_mean:.2f} +/- {off_rmse_std:.2f}")

    # Persist raw + summary results for inclusion in the report.
    pd.DataFrame({
        "repeat": range(1, N_REPEATS + 1),
        "ours_nll": results["ours"]["nll"], "ours_rmse": results["ours"]["rmse"], "ours_M": results["ours"]["M"],
        "official_nll": results["official"]["nll"], "official_rmse": results["official"]["rmse"],
        "official_M": results["official"]["M"],
    }).to_csv("results_raw.csv", index=False)

    with open("results_summary.txt", "w") as f:
        f.write("Concrete Compressive Strength (N=1030), 20 repeats, NGBoost reproduction\n")
        f.write("=" * 78 + "\n")
        f.write(f"{'Method':<28}{'Test NLL':<20}{'Test RMSE':<20}\n")
        f.write(f"{'Duan et al. (2020) paper':<28}{PAPER_NLL_MEAN:.2f} +/- {PAPER_NLL_STD:.2f}      "
                f"{PAPER_RMSE_MEAN:.2f} +/- {PAPER_RMSE_STD:.2f}\n")
        f.write(f"{'Ours (from scratch)':<28}{ours_nll_mean:.2f} +/- {ours_nll_std:.2f}      "
                f"{ours_rmse_mean:.2f} +/- {ours_rmse_std:.2f}\n")
        f.write(f"{'Official ngboost package':<28}{off_nll_mean:.2f} +/- {off_nll_std:.2f}      "
                f"{off_rmse_mean:.2f} +/- {off_rmse_std:.2f}\n")
        f.write(f"\nTotal run time: {elapsed:.1f} sec\n")

    print("\nSaved: results_raw.csv, results_summary.txt")


if __name__ == "__main__":
    main()

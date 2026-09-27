HTIN5005 Assignment 1 (2026) – submitted coursework by kmorris10. This is the official submission repo. Reuse of this code in another student's submission would breach academic integrity policy:  Part B -- Run Instructions
====================================================

Base paper: Avati, A., et al. (2020). "A Model is Not Enough: A Case of
AI-Enabled Palliative Care Delivery."
Reimplemented algorithm: NGBoost (Natural Gradient Boosting), from
Duan, T., Avati, A., Ding, D. Y., Thai, K. K., Basu, S., Ng, A., & Schuler,
A. (2020). NGBoost: Natural Gradient Boosting for Probabilistic Prediction.
ICML 2020, PMLR 108/119. arXiv:1910.03225.
Official reference code (consulted, not copied): github.com/stanfordmlgroup/ngboost

FILES
-----
ngboost_scratch.py       Our from-scratch NGBoost implementation
                          (NGBoostNormal class). This is the required
                          reimplementation of the paper's core algorithm.
run_experiment.py         Reproduces the paper's Table 1 (NLL) / Table 3
                          (RMSE) result on the UCI "Concrete Compressive
                          Strength" dataset, 20 repeats, comparing our
                          from-scratch model against the official `ngboost`
                          package under an identical protocol. Reported as
                          Table B1 in the accompanying report.
run_bonus_experiment.py   [Bonus] Tests a proposed improvement -- swapping
                          the Normal output distribution for a Log-Normal
                          one -- using the same 20-repeat protocol. Reported
                          as Table B2 in the accompanying report.
Concrete_Data.csv         Dataset (UCI "Concrete Compressive Strength",
                          N=1030, 8 features). Downloaded from a
                          GitHub-hosted mirror of the UCI file:
                          https://raw.githubusercontent.com/HansenHan/ML_UCI_Concrete_Compressive_Strength/main/Concrete_Data.csv
                          (original source: https://archive.ics.uci.edu/dataset/165)
results_raw.csv           Per-repeat NLL/RMSE for both models (core experiment).
results_summary.txt       Mean +/- std summary table (core experiment).
results_bonus.csv         Per-repeat paired NLL/RMSE, Normal vs Log-Normal (bonus).
fig1_validation_curve.png Validation NLL vs. boosting stage for one repeat,
                          illustrating the model-selection (M*) procedure.
fig2_comparison.png       Bar chart: paper vs. our reimplementation vs. the
                          official package, NLL and RMSE.
requirements.txt          Exact package versions used.

SETUP
-----
1. Create and activate a virtual environment (recommended -- avoids
   conflicts with system packages):
       python3 -m venv venv
       source venv/bin/activate        (Windows: venv\Scripts\activate)

2. Install dependencies:
       pip install -r requirements.txt

   Note: if `pip install ngboost` fails while building the `autograd-gamma`
   dependency with an "install_layout" / distutils error, this is a known
   issue with old packages under newer setuptools on some Linux distros
   (e.g. Debian-patched Python). Installing inside a clean virtual
   environment (as above) resolves it, because the venv's setuptools is
   not the OS-patched one.

RUNNING
-------
Run the scripts in this order from inside this folder:

    python3 run_experiment.py
        -> Reproduces the core experiment (Table 1 / Table 3 comparison in
           the original paper; reported as Table B1 in our report).
           Prints per-repeat results, then a summary table, and writes
           results_raw.csv + results_summary.txt.
           Expected run time: ~5-6 minutes (20 repeats x 2 models).

    python3 run_bonus_experiment.py
        -> Runs the bonus Log-Normal-vs-Normal comparison (reported as
           Table B2 in our report). Requires results_raw.csv from the
           previous step (reuses its splits/seeds for a fair paired
           comparison). Writes results_bonus.csv.
           Expected run time: ~3 minutes.

Figures were generated with short standalone snippets using matplotlib
(Agg backend); see the report's Methods/Experiments sections for the exact
code used to produce fig1_validation_curve.png and fig2_comparison.png,
built directly from the CSV outputs above and from ngboost_scratch.py.

HARDWARE
--------
Developed and timed on a standard single-CPU-core cloud container (no GPU
required). Total combined run time for both scripts is well under 10
minutes.

CITATION / ACADEMIC INTEGRITY NOTE
-----------------------------------
ngboost_scratch.py is an independent implementation written for this
assignment. It is NOT copied from the official `ngboost` package; the
official package's public API (not its source) was used only to
cross-check our natural-gradient formula and to serve as an independent
reference model in run_experiment.py. See the module docstring in
ngboost_scratch.py for the full citation.

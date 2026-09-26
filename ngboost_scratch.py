"""
ngboost_scratch.py

A from-scratch reimplementation of NGBoost (Natural Gradient Boosting) for
probabilistic regression, specialised to a Normal output distribution with
the logarithmic (MLE) scoring rule -- exactly the configuration used for the
UCI regression benchmarks in the original paper's Table 1 and Table 3.

SOURCE / CITATION
-----------------
This is an independent implementation written by the student for
HTIN5005 Assignment 1, Part B. It follows Algorithm 1 and the analysis in:

    Duan, T., Avati, A., Ding, D. Y., Thai, K. K., Basu, S., Ng, A., &
    Schuler, A. (2020). NGBoost: Natural Gradient Boosting for Probabilistic
    Prediction. Proceedings of the 37th International Conference on Machine
    Learning (ICML), PMLR 108/119. arXiv:1910.03225.

The official reference implementation released by the authors,
    https://github.com/stanfordmlgroup/ngboost
was consulted (read only, not copied) to confirm two design choices that are
not fully spelled out by closed-form algebra alone:
    1. the successive-halving line-search procedure for the per-stage scaling
       factor rho (paper, Section 3.4), and
    2. the convention of scaling the natural gradient step by an additional
       learning rate eta on top of rho (paper, Algorithm 1).
No source code from that repository is copied into this file; the natural
gradient formula for the Normal distribution below is derived independently
in the accompanying report (Methods section) from the Fisher information of
a Normal(mu, log(sigma)) parameterisation.

The official `ngboost` PyPI package (by the same authors) is used elsewhere
in this project (see run_experiment.py) purely as an independent reference
to sanity-check that this from-scratch implementation reproduces sensible
numbers -- it is not used to produce the "our reimplementation" results.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class NGBoostNormal:
    """
    Natural Gradient Boosting for a Normal(mu, sigma) output distribution,
    fit by minimising the average logarithmic (MLE) score, i.e. the negative
    log-likelihood, using natural-gradient multiparameter boosting
    (Duan et al., 2020, Algorithm 1).

    Parameters
    ----------
    n_estimators : int
        Maximum number of boosting stages M.
    learning_rate : float
        Global learning rate eta applied on top of the per-stage line-search
        scale rho (paper default: 0.01 for all UCI datasets except Year MSD).
    max_depth : int
        Max depth of each per-parameter regression-tree base learner
        (paper default: 3).
    min_samples_leaf : int
        Passed through to the underlying DecisionTreeRegressor.
    """

    def __init__(self, n_estimators=800, learning_rate=0.01, max_depth=3,
                 min_samples_leaf=1, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Core distributional machinery for Normal(mu, log_sigma)
    # ------------------------------------------------------------------
    @staticmethod
    def _neg_log_likelihood(mu, log_sigma, y):
        """L(theta, y) = -log P_theta(y) for y ~ N(mu, sigma^2)."""
        sigma = np.exp(log_sigma)
        return log_sigma + 0.5 * np.log(2 * np.pi) + 0.5 * ((y - mu) / sigma) ** 2

    @staticmethod
    def _natural_gradient(mu, log_sigma, y):
        """
        Natural gradient of the log score w.r.t. theta = (mu, log_sigma).

        Derivation (see report Methods section): the Fisher information of a
        Normal(mu, log_sigma) parameterisation is the diagonal matrix
        I(theta) = diag(1/sigma^2, 2), so the natural gradient
        I(theta)^-1 grad(L) simplifies in closed form to:
            g_mu       = (mu - y)
            g_log_sigma = 0.5 * (1 - ((y - mu) / sigma)^2)
        This matches the analytic form used by the original NGBoost package
        for the Normal distribution, confirmed by inspection of its public
        API documentation (not its source).
        """
        sigma = np.exp(log_sigma)
        g_mu = (mu - y)
        g_log_sigma = 0.5 * (1.0 - ((y - mu) / sigma) ** 2)
        return g_mu, g_log_sigma

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit the boosting ensemble. If X_val/y_val are provided, validation
        NLL is tracked at every stage (self.val_nll_) so the caller can pick
        the best number of stages, exactly as in the paper's model-selection
        protocol (Section 4).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)

        # theta^(0): fit the marginal distribution (paper, Algorithm 1)
        mu0 = float(np.mean(y))
        log_sigma0 = float(np.log(np.std(y) + 1e-6))
        self.init_params_ = (mu0, log_sigma0)

        mu = np.full(n, mu0)
        log_sigma = np.full(n, log_sigma0)

        if X_val is not None:
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float)
            mu_val = np.full(len(y_val), mu0)
            log_sigma_val = np.full(len(y_val), log_sigma0)
            self.val_nll_ = [self._neg_log_likelihood(mu_val, log_sigma_val, y_val).mean()]

        self.trees_mu_ = []
        self.trees_logsigma_ = []
        self.rhos_ = []

        rng_state = self.random_state
        for m in range(self.n_estimators):
            g_mu, g_log_sigma = self._natural_gradient(mu, log_sigma, y)

            # One shallow regression tree per distributional parameter,
            # each fit to predict (the negative of) its natural-gradient
            # component -- i.e. the direction of steepest descent.
            tree_mu = DecisionTreeRegressor(
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                random_state=None if rng_state is None else rng_state + m,
            )
            tree_logsigma = DecisionTreeRegressor(
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                random_state=None if rng_state is None else rng_state + m + 100000,
            )
            tree_mu.fit(X, g_mu)
            tree_logsigma.fit(X, g_log_sigma)

            f_mu = tree_mu.predict(X)
            f_logsigma = tree_logsigma.predict(X)

            # Line search for rho via successive halving (paper, Sec 3.4):
            # start at rho=1 and halve until the update does not increase
            # the total training loss relative to the current iterate.
            rho = 1.0
            current_loss = self._neg_log_likelihood(mu, log_sigma, y).mean()
            for _ in range(30):  # 30 halvings -> rho as small as ~1e-9
                new_mu = mu - rho * f_mu
                new_log_sigma = log_sigma - rho * f_logsigma
                new_loss = self._neg_log_likelihood(new_mu, new_log_sigma, y).mean()
                if np.isfinite(new_loss) and new_loss <= current_loss:
                    break
                rho *= 0.5

            # Apply the (learning-rate-scaled) update.
            mu = mu - self.learning_rate * rho * f_mu
            log_sigma = log_sigma - self.learning_rate * rho * f_logsigma

            self.trees_mu_.append(tree_mu)
            self.trees_logsigma_.append(tree_logsigma)
            self.rhos_.append(rho)

            if X_val is not None:
                f_mu_val = tree_mu.predict(X_val)
                f_logsigma_val = tree_logsigma.predict(X_val)
                mu_val = mu_val - self.learning_rate * rho * f_mu_val
                log_sigma_val = log_sigma_val - self.learning_rate * rho * f_logsigma_val
                self.val_nll_.append(self._neg_log_likelihood(mu_val, log_sigma_val, y_val).mean())

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_dist(self, X, n_stages=None):
        """Return (mu, sigma) arrays using the first n_stages trees (all by default)."""
        X = np.asarray(X, dtype=float)
        mu0, log_sigma0 = self.init_params_
        n = X.shape[0]
        mu = np.full(n, mu0)
        log_sigma = np.full(n, log_sigma0)

        stages = len(self.trees_mu_) if n_stages is None else n_stages
        for m in range(stages):
            f_mu = self.trees_mu_[m].predict(X)
            f_logsigma = self.trees_logsigma_[m].predict(X)
            rho = self.rhos_[m]
            mu = mu - self.learning_rate * rho * f_mu
            log_sigma = log_sigma - self.learning_rate * rho * f_logsigma
        return mu, np.exp(log_sigma)

    def predict(self, X, n_stages=None):
        """Point prediction: the predicted mean E[y|x]."""
        mu, _ = self.predict_dist(X, n_stages=n_stages)
        return mu

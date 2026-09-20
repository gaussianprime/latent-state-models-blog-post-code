"""Shared fitting and scoring for the fresh-parity experiments.

All exact models are compared on log p(x[1:] | x[0]), with the same
denominator. Restarts are selected using training likelihood only. A failed
fit raises rather than becoming an infinite improvement for its competitor.
The older GPB2 SLDS is retained as an explicitly approximate sensitivity.
"""

from dataclasses import dataclass, field
import time
import warnings

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM


WEIGHT_COLUMNS = ["l1", "l2", "trace", "spectral", "code_sparsity",
                  "computational_sparsity", "mean_lambda", "variance_lambda",
                  "mean_w", "median_w", "var_w", "mean_b", "median_b", "var_b"]
FUNCTIONAL_COLUMNS = ["loss_gap", "accuracy_gap", "ever_learned_fraction",
                      "forgetting_events_per_example", "train_margin_mean",
                      "eval_margin_mean", "train_margin_positive_fraction",
                      "eval_margin_positive_fraction"]
SUBSETS = {"weight": WEIGHT_COLUMNS, "functional": FUNCTIONAL_COLUMNS,
           "weight+functional": WEIGHT_COLUMNS + FUNCTIONAL_COLUMNS}
EXACT_MODELS = ["gmm", "hmm", "lds", "var1", "var1_full", "arhmm_diag", "arhmm_full"]


def zscore_run(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("A trajectory must have at least two finite observations")
    sd = x.std(axis=0)
    return (x - x.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def raw_features(runs):
    """Compute summaries BEFORE per-trajectory centering removes the mean.

    The regression's cross-validation pipeline scales these features using
    its training fold only. There is no learned preprocessing here.
    """
    return np.stack([np.r_[r.mean(0), r.std(0)] for r in runs])


def forward_increments(log_pi, log_transitions, log_emissions):
    """Exact per-observation HMM predictive log likelihoods in log space."""
    out = np.empty(len(log_emissions))
    alpha = log_pi + log_emissions[0]
    out[0] = logsumexp(alpha)
    alpha -= out[0]
    for t in range(1, len(out)):
        trans = (log_transitions if log_transitions.ndim == 2
                 else log_transitions[min(t - 1, len(log_transitions) - 1)])
        alpha = logsumexp(alpha[:, None] + trans, axis=0) + log_emissions[t]
        out[t] = logsumexp(alpha)
        alpha -= out[t]
    return out


@dataclass
class FittedModel:
    kind: str
    k: int
    d: int
    model: object = None
    params: object = None
    metadata: dict = field(default_factory=dict)
    quantizer: object = None

    def joint_loglik(self, x):
        if self.kind == "gmm":
            return float(self.model.score_samples(x).sum())
        if self.kind == "hmm":
            return float(self.model.score(x))
        if self.kind.startswith("arhmm"):
            return float(self.model.log_likelihood([x]))
        if self.kind == "lds":
            import jax.numpy as jnp
            return float(self.model.marginal_log_prob(self.params, jnp.asarray(x)))
        if self.kind == "lds_custom":
            from src.nulls.lds import lds_score
            return float(lds_score(self.params, x, [len(x)]))
        if self.kind == "slds_approx":
            from src.nulls.slds import slds_score
            return float(slds_score(self.params, x, [len(x)]))
        # VAR defines the conditional distribution directly. Its dummy
        # initial contribution is zero and never included in comparisons.
        return float(self.step_scores(x)[1:].sum())

    def conditional_loglik(self, x):
        if self.kind.startswith("var1"):
            value = self.step_scores(x)[1:].sum()
        else:
            value = self.joint_loglik(x) - self.joint_loglik(x[:1])
        if not np.isfinite(value):
            raise FloatingPointError(f"{self.kind}: non-finite conditional score")
        return float(value)

    def step_scores(self, x):
        if self.kind == "gmm":
            return self.model.score_samples(x)
        if self.kind == "hmm":
            with np.errstate(divide="ignore"):
                return forward_increments(np.log(self.model.startprob_),
                    np.log(self.model.transmat_), self.model._compute_log_likelihood(x))
        if self.kind.startswith("arhmm"):
            inputs = np.zeros((len(x), 0))
            mask = np.ones_like(x, dtype=bool)
            emissions = np.asarray(self.model.observations.log_likelihoods(x, inputs, mask, None))
            transitions = np.asarray(self.model.transitions.log_transition_matrices(x, inputs, mask, None))
            pi = np.asarray(self.model.init_state_distn.log_initial_state_distn)
            return forward_increments(pi, transitions, emissions)
        if self.kind.startswith("var1"):
            weights, covariance = self.params
            residual = x[1:] - np.column_stack([x[:-1], np.ones(len(x) - 1)]) @ weights
            out = np.zeros(len(x))
            if self.kind == "var1":
                out[1:] = (-0.5 * (np.log(2 * np.pi * covariance) + residual**2 / covariance)).sum(1)
            else:
                out[1:] = multivariate_normal.logpdf(residual, np.zeros(self.d), covariance)
            return out
        if self.kind == "lds":
            import jax.numpy as jnp
            p = self.params
            posterior = self.model.filter(p, jnp.asarray(x))
            means = np.asarray(posterior.filtered_means)
            covs = np.asarray(posterior.filtered_covariances)
            A, Q = np.asarray(p.dynamics.weights), np.asarray(p.dynamics.cov)
            C, R = np.asarray(p.emissions.weights), np.asarray(p.emissions.cov)
            out = np.empty(len(x))
            for t in range(len(x)):
                if t == 0:
                    mean, cov = np.asarray(p.initial.mean), np.asarray(p.initial.cov)
                else:
                    mean = A @ means[t - 1] + np.asarray(p.dynamics.bias)
                    cov = A @ covs[t - 1] @ A.T + Q
                emission_cov = C @ cov @ C.T + R
                emission_cov = (emission_cov + emission_cov.T) / 2
                out[t] = multivariate_normal.logpdf(x[t], C @ mean + np.asarray(p.emissions.bias),
                                                    emission_cov, allow_singular=False)
            return out
        raise ValueError(f"Stepwise decomposition unavailable for {self.kind}")

    def states(self, x):
        if self.kind in ("gmm", "hmm"):
            return self.model.predict(x)
        if self.kind.startswith("arhmm"):
            return np.asarray(self.model.most_likely_states(x), dtype=int)
        raise ValueError(f"Discrete readout unavailable for {self.kind}")

    def latent(self, x):
        import jax.numpy as jnp
        return np.asarray(self.model.smoother(self.params, jnp.asarray(x)).smoothed_means)

    def features(self, runs, training_runs=None):
        if self.kind in ("gmm", "hmm") or self.kind.startswith("arhmm"):
            return {self.kind: np.stack([np.bincount(self.states(x), minlength=self.k) / len(x)
                                         for x in runs])}
        if self.kind == "lds":
            if self.quantizer is None:
                if training_runs is None:
                    raise ValueError("Fit the latent quantizer on training runs first")
                training_latents = np.vstack([self.latent(x) for x in training_runs])
                self.quantizer = KMeans(self.k, n_init=10, random_state=0).fit(training_latents)
            zs = [self.latent(x) for x in runs]
            return {
                "lds": np.stack([z.mean(0) for z in zs]),
                "lds_ms": np.stack([np.r_[z.mean(0), z.std(0)] for z in zs]),
                "lds_occ": np.stack([np.bincount(self.quantizer.predict(z), minlength=self.k) / len(z)
                                      for z in zs]),
            }
        raise ValueError(f"No regression features for {self.kind}")


def _fit_once(kind, runs, k, seed, n_iter):
    d = runs[0].shape[1]
    data, lengths = np.vstack(runs), [len(x) for x in runs]
    result = FittedModel(kind, k, d)
    if kind == "gmm":
        result.model = GaussianMixture(k, covariance_type="diag", n_init=1,
            max_iter=n_iter, reg_covar=1e-6, random_state=seed).fit(data)
        result.metadata["converged"] = bool(result.model.converged_)
    elif kind == "hmm":
        result.model = GaussianHMM(k, covariance_type="diag", n_iter=n_iter,
            min_covar=1e-6, random_state=seed).fit(data, lengths)
        result.metadata["converged"] = bool(result.model.monitor_.converged)
    elif kind.startswith("var1"):
        target = np.vstack([x[1:] for x in runs])
        design = np.vstack([np.column_stack([x[:-1], np.ones(len(x) - 1)]) for x in runs])
        weights = np.linalg.lstsq(design, target, rcond=None)[0]
        residual = target - design @ weights
        if kind == "var1":
            cov = np.maximum((residual**2).mean(0), 1e-6)
        else:
            cov = residual.T @ residual / len(residual)
            vals, vecs = np.linalg.eigh((cov + cov.T) / 2)
            cov = (vecs * np.maximum(vals, 1e-6)) @ vecs.T
        result.params = (weights, cov)
    elif kind.startswith("arhmm"):
        import ssm
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            result.model = ssm.HMM(k, d, observations="diagonal_ar" if kind == "arhmm_diag" else "ar")
            history = result.model.fit(runs, method="em", num_iters=n_iter, verbose=0)
            result.metadata["fit_history_last"] = float(history[-1])
        finally:
            np.random.set_state(state)
    elif kind == "lds":
        from jax import config
        config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import jax.random as jr
        from jax.tree_util import tree_leaves
        from src.lds_numerics import SymmetricLinearGaussianConjugateSSM, validate_lds_covariances
        if len(set(lengths)) != 1:
            raise ValueError("The library LDS batch requires equal-length training runs")
        result.model = SymmetricLinearGaussianConjugateSSM(state_dim=k, emission_dim=d)
        params, props = result.model.initialize(jr.PRNGKey(seed))
        result.params, history = result.model.fit_em(params, props, jnp.asarray(np.stack(runs)),
                                                     num_iters=n_iter, verbose=False)
        if not all(np.isfinite(np.asarray(x)).all() for x in tree_leaves(result.params)):
            raise FloatingPointError("LDS fit produced non-finite parameters")
        result.metadata["covariances"] = validate_lds_covariances(result.params)
        result.metadata["covariance_update"] = "symmetric_map_v1"
        result.metadata["fit_history_last"] = float(history[-1])
    elif kind == "lds_custom":
        from src.nulls.lds import em_fit
        result.params, history, converged = em_fit(data, lengths, k, seed=seed, n_init=1,
                                                   max_iter=n_iter)
        result.metadata.update(converged=bool(converged), fit_history_last=float(history[-1]))
    elif kind == "slds_approx":
        from src.nulls.slds import slds_em_fit
        result.params, history, converged = slds_em_fit(data, lengths, latent_dim=3,
            n_modes=k, seed=seed, n_init=1, max_iter=n_iter)
        result.metadata.update(converged=bool(converged), fit_history_last=float(history[-1]))
    else:
        raise ValueError(f"Unknown model: {kind}")
    return result


def fit_model(kind, training_runs, k=3, seed=0, n_iter=60, n_starts=2):
    """Only training data is accepted: test scores cannot select a restart."""
    started = time.monotonic()
    best, best_ll = None, -np.inf
    attempts = []
    for restart in range(1 if kind.startswith("var1") else n_starts):
        fit_seed = seed + 10000 * restart
        try:
            with warnings.catch_warnings(record=True) as caught:
                model = _fit_once(kind, training_runs, k, fit_seed, n_iter)
                ll = sum(model.joint_loglik(x) for x in training_runs)
            if not np.isfinite(ll):
                raise FloatingPointError("Non-finite training likelihood")
            attempts.append(dict(seed=fit_seed, train_loglik=float(ll),
                                 warnings=[str(w.message) for w in caught]))
            if ll > best_ll:
                best, best_ll = model, ll
                best.metadata["selected_seed"] = fit_seed
        except Exception as exc:
            attempts.append(dict(seed=fit_seed, error=f"{type(exc).__name__}: {exc}"))
    if best is None:
        raise RuntimeError(f"All {kind} restarts failed: {attempts}")
    best.metadata.update(attempts=attempts, train_loglik=float(best_ll),
        seconds=time.monotonic() - started,
        n_iter=n_iter, n_starts=n_starts, k=k, n_features=best.d,
        likelihood_kind="GPB2_approximation" if kind == "slds_approx" else "exact",
        restart_selection="training_joint_loglik")
    return best

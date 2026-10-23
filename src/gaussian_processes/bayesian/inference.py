"""Muestreo, diagnósticos y transformación de muestras en probabilidades."""

import time

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO
from numpyro.infer import MCMC, NUTS
from scipy.sparse import csr_matrix
from scipy.special import softmax

from .design import CLASS_CONTRASTS, basis_scales
from .model import categorical_gp


def sample_posterior(basis, degrees, factor, labels, settings, seed):
    """NUTS con cuatro cadenas por defecto, doble precisión y warmup descartado."""
    started = time.perf_counter()
    kernel = NUTS(
        categorical_gp,
        target_accept_prob=settings["target_accept"],
        max_tree_depth=settings["max_tree_depth"],
    )
    sampler = MCMC(
        kernel,
        num_warmup=settings["warmup"],
        num_samples=settings["draws"],
        num_chains=settings["chains"],
        chain_method="sequential",
        progress_bar=True,
    )
    print(
        f"NUTS: {settings['chains']} cadenas; {settings['warmup']} warmup + {settings['draws']} muestras por cadena",
        flush=True,
    )
    sampler.run(
        jax.random.PRNGKey(seed),
        jnp.asarray(basis),
        jnp.asarray(degrees),
        BCOO.from_scipy_sparse(csr_matrix(factor)),
        settings,
        jnp.asarray(labels),
        extra_fields=("diverging", "num_steps", "accept_prob"),
    )
    samples = {
        key: np.asarray(value)
        for key, value in sampler.get_samples(group_by_chain=True).items()
    }
    extras = {
        key: np.asarray(value)
        for key, value in sampler.get_extra_fields(group_by_chain=True).items()
    }
    diagnostics = diagnose(samples, extras, settings)
    diagnostics["seconds"] = time.perf_counter() - started
    return samples, extras, diagnostics


def diagnose(samples, extras, settings):
    """R-hat de rango y ESS bulk/tail de todos los parámetros, incluidos latentes."""
    tree = az.from_dict({"posterior": samples})
    summary = az.summary(tree, kind="diagnostics", round_to=8)
    hyper = az.summary(
        tree,
        var_names=["sigma_spatial", "sigma_phylogenetic", "smoothing", "intercept"],
        round_to=8,
    )
    maximum = float(summary.r_hat.max())
    bulk, tail = float(summary.ess_bulk.min()), float(summary.ess_tail.min())
    divergences = int(extras["diverging"].sum())
    depth_hits = int((extras["num_steps"] >= 2 ** settings["max_tree_depth"] - 1).sum())
    finite = bool(np.isfinite(summary[["r_hat", "ess_bulk", "ess_tail"]]).all().all())
    passed = (
        finite
        and maximum < 1.01
        and min(bulk, tail) >= 400
        and divergences == 0
        and depth_hits == 0
    )
    return {
        "passed": bool(passed),
        "max_rhat": maximum,
        "min_ess_bulk": bulk,
        "min_ess_tail": tail,
        "divergences": divergences,
        "tree_depth_hits": depth_hits,
        "mean_acceptance": float(extras["accept_prob"].mean()),
        "criteria": "Rhat<1.01; ESS bulk/tail>=400; cero divergencias y saturaciones de profundidad",
        "hyperparameters": hyper.reset_index().to_dict("records"),
        "worst_rhat": summary.nlargest(5, "r_hat").reset_index().to_dict("records"),
    }


def flatten_samples(samples, maximum):
    """Subselección determinista equilibrada entre cadenas para predecir."""
    total = (
        next(iter(samples.values())).shape[0] * next(iter(samples.values())).shape[1]
    )
    indices = np.linspace(0, total - 1, min(maximum, total), dtype=int)
    return {
        key: value.reshape((-1, *value.shape[2:]))[indices]
        for key, value in samples.items()
    }


def probability_draws(samples, basis, degrees, conditional_mean, residual_factor, seed):
    """Muestras conjuntas: parámetros + funciones + residual condicional nuevo.

    Devuelve [muestra, lengua, clase]. La correlación entre faltantes se conserva
    mediante el factor residual completo; softmax se aplica ANTES de promediar.
    """
    rng = np.random.default_rng(seed)
    count = len(samples["sigma_spatial"])
    probabilities = np.empty((count, len(basis), 3))
    for draw in range(count):
        spatial = samples["sigma_spatial"][draw] * (
            basis
            @ (
                basis_scales(degrees, samples["smoothing"][draw])[:, None]
                * samples["z_spatial"][draw]
            )
        )
        phylogenetic = samples["sigma_phylogenetic"][draw] * (
            conditional_mean @ samples["z_phylogenetic"][draw]
            + residual_factor @ rng.normal(size=(residual_factor.shape[1], 2))
        )
        logits = (
            samples["intercept"][draw] + spatial + phylogenetic
        ) @ CLASS_CONTRASTS.T
        probabilities[draw] = softmax(logits, axis=-1)
    return probabilities


def prior_predictive(basis, degrees, factor, settings):
    """Simula sin etiquetas; resume frecuencias esperadas y confianza del prior."""
    rng = np.random.default_rng(settings["seed"])
    n = settings["prior_draws"]
    draws = {
        "sigma_spatial": abs(rng.normal(0, settings["amplitude_scale"], n)),
        "sigma_phylogenetic": abs(rng.normal(0, settings["amplitude_scale"], n)),
        "smoothing": rng.lognormal(
            np.log(settings["smoothing_median"]), settings["smoothing_log_sd"], n
        ),
        "intercept": rng.normal(0, settings["intercept_scale"], (n, 2)),
        "z_spatial": rng.normal(size=(n, basis.shape[1], 2)),
        "z_phylogenetic": rng.normal(size=(n, factor.shape[1], 2)),
    }
    p = probability_draws(
        draws, basis, degrees, factor, np.empty((len(basis), 0)), settings["seed"] + 1
    )
    return {
        "class_order": ["OV", "Sin dominante", "VO"],
        "expected_class_fraction_quantiles_025_50_975": np.quantile(
            p.mean(axis=1), [0.025, 0.5, 0.975], axis=0
        ).tolist(),
        "mean_max_probability": float(p.max(axis=-1).mean()),
        "fraction_probability_above_095": float((p.max(axis=-1) > 0.95).mean()),
    }

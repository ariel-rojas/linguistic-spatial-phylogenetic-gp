"""Modelo categórico con priors propios y parametrización no centrada."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .design import CLASS_CONTRASTS


def categorical_gp(spatial_basis, degrees, phylogenetic_factor, settings, labels=None):
    """Declara la posterior; solo labels observados aportan verosimilitud.

    Dos contrastes generan tres logits de suma cero y prior simétrico respecto
    de las clases. Los hiperparámetros se comparten entre contrastes. El árbol,
    las familias y el grado máximo permanecen fijos.
    """
    sigma_spatial = numpyro.sample(
        "sigma_spatial", dist.HalfNormal(settings["amplitude_scale"])
    )
    sigma_phylogenetic = numpyro.sample(
        "sigma_phylogenetic", dist.HalfNormal(settings["amplitude_scale"])
    )
    smoothing = numpyro.sample(
        "smoothing",
        dist.LogNormal(
            jnp.log(settings["smoothing_median"]), settings["smoothing_log_sd"]
        ),
    )
    intercept = numpyro.sample(
        "intercept",
        dist.Normal(0, settings["intercept_scale"]).expand((2,)).to_event(1),
    )
    z_spatial = numpyro.sample(
        "z_spatial", dist.Normal(0, 1).expand((spatial_basis.shape[1], 2)).to_event(2)
    )
    z_phylogenetic = numpyro.sample(
        "z_phylogenetic",
        dist.Normal(0, 1).expand((phylogenetic_factor.shape[1], 2)).to_event(2),
    )
    weights = 1 / (1 + smoothing * (degrees * (degrees + 1)) ** 2)
    scales = jnp.sqrt(weights / weights.sum())
    spatial = sigma_spatial * (spatial_basis @ (scales[:, None] * z_spatial))
    phylogenetic = sigma_phylogenetic * (phylogenetic_factor @ z_phylogenetic)
    logits = (intercept + spatial + phylogenetic) @ jnp.asarray(CLASS_CONTRASTS).T
    with numpyro.plate("languages", spatial_basis.shape[0]):
        numpyro.sample("observed", dist.Categorical(logits=logits), obs=labels)

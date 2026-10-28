"""Modelo categórico con priors propios y parametrización no centrada."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .design import CLASS_CONTRASTS


def categorical_gp(
    spatial_basis,
    degrees,
    phylogenetic_factor,
    settings,
    labels=None,
):
    """Declara la posterior; solo labels observados aportan verosimilitud.

    Dos contrastes generan tres logits de suma cero y prior simétrico respecto
    de las clases. Los hiperparámetros se comparten entre contrastes. El grado
    máximo efectivo tiene un prior uniforme continuo; un corte logístico
    diferenciable permite estimarlo junto con los parámetros restantes usando
    NUTS, sin un paso Gibbs discreto que puede quedar atrapado en un candidato.
    """
    max_degree = numpyro.sample(
        "max_degree",
        dist.Uniform(settings["degree_min"], settings["degree_max"]),
    )
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
    degree_weights = jax.nn.sigmoid(
        (max_degree + 0.5 - jnp.asarray(degrees))
        / settings["degree_taper_width"]
    )
    raw_weights = 1 / (1 + smoothing * (degrees * (degrees + 1)) ** 2)
    weights = degree_weights * raw_weights
    scales = jnp.sqrt(weights / weights.sum())
    spatial = sigma_spatial * (spatial_basis @ (scales[:, None] * z_spatial))
    phylogenetic = sigma_phylogenetic * (phylogenetic_factor @ z_phylogenetic)
    # Cada fila suma intercepto, espacio y filogenia en dos coordenadas latentes.
    # CLASS_CONTRASTS.T tiene forma (2, 3): la proyección da un logit por clase.
    # Dos coordenadas bastan porque softmax no cambia al sumar la misma constante
    # a los tres logits; la matriz de contrastes fija esa dirección común en cero.
    logits = (intercept + spatial + phylogenetic) @ jnp.asarray(CLASS_CONTRASTS).T
    with numpyro.plate("languages", spatial_basis.shape[0]):
        numpyro.sample("observed", dist.Categorical(logits=logits), obs=labels)

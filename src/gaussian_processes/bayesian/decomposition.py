"""Predicción pareada para describir sensibilidad a cada componente del GP."""

import numpy as np
from scipy.special import softmax

from .design import CLASS_CONTRASTS, basis_scales


def predictive_components(
    samples, basis, degrees, conditional_mean, residual_factor, seed,
    degree_taper_width=0.25,
):
    """Devuelve p completa y dos contrafactuales que omiten un efecto cada uno.

    La puntuación de influencia es la distancia de variación total
    TV(p, p_sin_componente) = 1/2 sum_c |p_c - p_sin_componente,c|.
    Las comparaciones son pareadas por muestra y usan el mismo residual aleatorio.
    Son sensibilidad descriptiva de esta predicción condicionada al ajuste, no
    porcentajes causales, reparto de varianza ni parámetros latentes estimados.
    """
    rng = np.random.default_rng(seed)
    count, n_targets = len(samples["sigma_spatial"]), len(basis)
    output = {
        key: np.empty((count, n_targets, 3))
        for key in ["full", "without_spatial", "without_phylogenetic"]
    }
    for draw in range(count):
        spatial = samples["sigma_spatial"][draw] * (
            basis
            @ (
                basis_scales(
                    degrees,
                    samples["smoothing"][draw],
                    samples["max_degree"][draw],
                    degree_taper_width,
                )[:, None]
                * samples["z_spatial"][draw]
            )
        )
        residual = residual_factor @ rng.normal(size=(residual_factor.shape[1], 2))
        phylogenetic = samples["sigma_phylogenetic"][draw] * (
            conditional_mean @ samples["z_phylogenetic"][draw] + residual
        )
        intercept = samples["intercept"][draw]
        for name, effect in [
            ("full", intercept + spatial + phylogenetic),
            ("without_spatial", intercept + phylogenetic),
            ("without_phylogenetic", intercept + spatial),
        ]:
            output[name][draw] = softmax(effect @ CLASS_CONTRASTS.T, axis=-1)
    return output


def total_variation(first, second):
    """Cambio máximo de probabilidad agregada entre dos distribuciones categóricas."""
    return 0.5 * np.abs(first - second).sum(axis=-1)

"""Contexto didáctico calculado desde la ejecución guardada, no desde el JSON actual."""

import numpy as np
from scipy.stats import halfnorm, lognorm, norm, uniform


def prior_summary(distribution, support, formula, logarithmic=False):
    """Cuantiles exactos y masas analíticas; el gráfico recorta 0,1–99,9 percentiles.

    El suavizado se representa en log10 para hacer visibles varios órdenes de
    magnitud. Las masas NO se renormalizan: se omite 0,2 % de las colas del prior.
    """
    limits = distribution.ppf([0.001, 0.999])
    edges = np.linspace(*(np.log10(limits) if logarithmic else limits), 21)
    original_edges = 10**edges if logarithmic else edges
    return {
        "support": support,
        "formula": formula,
        "interval": distribution.ppf([0.025, 0.975]).tolist(),
        "median": float(distribution.median()),
        "logarithmic": logarithmic,
        "chart": {
            "edges": edges.tolist(),
            "mass": np.diff(distribution.cdf(original_edges)).tolist(),
            "mean": float(
                np.log10(distribution.median()) if logarithmic else distribution.mean()
            ),
            "interval": (
                np.log10(distribution.ppf([0.025, 0.975]))
                if logarithmic
                else distribution.ppf([0.025, 0.975])
            ).tolist(),
        },
    }


def build_bayesian_context(provenance):
    """Reconstruye los priors de categorical_gp con hiperparámetros archivados."""
    s = provenance["sampling"]
    amplitude = s["amplitude_scale"]
    intercept = s["intercept_scale"]
    median, log_sd = s["smoothing_median"], s["smoothing_log_sd"]
    return {
        "priors": {
            "Grado máximo efectivo L": prior_summary(
                uniform(loc=s["degree_min"], scale=s["degree_max"] - s["degree_min"]),
                f"{s['degree_min']} ≤ L ≤ {s['degree_max']}",
                f"L ~ Uniforme({s['degree_min']}, {s['degree_max']})",
            ),
            "Amplitudes espacial y filogenética (cada una)": prior_summary(
                halfnorm(scale=amplitude),
                "σ ≥ 0; sin máximo finito",
                f"σ ~ HalfNormal(escala={amplitude})",
            ),
            "Suavizado espacial": prior_summary(
                lognorm(s=log_sd, scale=median),
                "λ > 0; sin máximo finito",
                f"log(λ) ~ Normal(log({median}), desviación={log_sd})",
                True,
            ),
            "Interceptos (cada uno de los dos)": prior_summary(
                norm(scale=intercept),
                "−∞ < α < +∞",
                f"αₖ ~ Normal(media=0, desviación={intercept})",
            ),
            "Coeficientes latentes (cada elemento de Z)": prior_summary(
                norm(), "−∞ < z < +∞", "z ~ Normal(media=0, desviación=1)"
            ),
        },
        "configuration": provenance["configuration"],
        "sampling": s,
        "degree_prior": {
            "type": "Uniforme continuo",
            "lower": s["degree_min"],
            "upper": s["degree_max"],
            "taper_width": s["degree_taper_width"],
        },
        "counts": {
            k: provenance[k] for k in ["n_train", "n_test", "n_final_fit", "n_imputed"]
        },
        "metrics": provenance["metrics"],
        "prior_predictive": provenance["prior_predictive"],
    }

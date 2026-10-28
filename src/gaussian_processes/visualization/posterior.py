"""Explorador HTML autónomo a partir de muestras guardadas, sin volver a ajustar."""

import json
from pathlib import Path

import numpy as np

from ..bayesian.decomposition import predictive_components, total_variation
from ..bayesian.design import phylogenetic_design, spherical_basis
from ..bayesian.inference import flatten_samples
from ..configuration import CLASSES, PROBABILITY_COLUMNS
from ..data.loading import load_data
from ..data.phylogeny import load_phylogeny


def distribution_summary(values, edges=None):
    """Histograma de masa por intervalo y resumen de TODAS las muestras recibidas."""
    values = np.asarray(values, dtype=float).ravel()
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Muestras vacías o no finitas")
    counts, edges = np.histogram(values, bins=20 if edges is None else edges)
    if counts.sum() != len(values):
        raise ValueError("Los intervalos no cubren todas las muestras")
    return {
        "edges": edges.tolist(),
        "mass": (counts / len(values)).tolist(),
        "mean": float(values.mean()),
        "interval": np.quantile(values, [0.025, 0.975]).tolist(),
        "n": len(values),
    }


def build_posterior_payload(folder, provenance, predictions):
    """Alinea por ID y valida las muestras contra CSV/configuración de la ejecución.

    Los parámetros son globales a cada ajuste. Las probabilidades sí son específicas
    de cada lengua; evaluation e imputation nunca se combinan como una posterior.
    Se exportan histogramas, no miles de muestras repetidas dentro de cada popup.
    """
    cases, stages = [], {}
    # El archivo histórico puede no tener aún los contrafactuales: reconstruirlos
    # desde las muestras guardadas conserva la ejecución sin volver a correr NUTS.
    eligible = data = covariance = basis = degrees = None
    rows = predictions.set_index("ID")
    for stage, role, diagnostic in [
        ("evaluation", "test", "evaluation_diagnostics"),
        ("imputation", "imputed", "imputation_diagnostics"),
    ]:
        with np.load(folder / f"{stage}.npz", allow_pickle=False) as archive:
            if archive["classes"].tolist() != CLASSES:
                raise ValueError("Orden de clases incompatible en muestras")
            if json.loads(str(archive["settings_json"])) != provenance["sampling"]:
                raise ValueError("Muestras y configuración de muestreo no coinciden")
            if json.loads(str(archive["diagnostics_json"])) != provenance[diagnostic]:
                raise ValueError("Muestras y diagnósticos no coinciden")
            ids, probabilities = archive["target_ids"], archive["probabilities"]
            if (
                "spatial_influence" not in archive
                or "phylogenetic_influence" not in archive
            ):
                if eligible is None:
                    data = load_data()
                    eligible = data.loc[data.elegible_modelo].reset_index(drop=True)
                    eligible["_design_index"] = np.arange(len(eligible))
                    covariance, _ = load_phylogeny(provenance["configuration"], data)
                    basis, degrees = spherical_basis(
                        eligible.Latitude,
                        eligible.Longitude,
                        provenance["sampling"]["degree_max"],
                    )
                by_id = eligible.set_index("ID")
                fit = by_id.loc[archive["fit_ids"]]
                target = by_id.loc[ids]
                _, mean, residual, _ = phylogenetic_design(
                    covariance,
                    fit.tree_tip_index.to_numpy(),
                    target.tree_tip_index.to_numpy(),
                )
                saved = {
                    key.removeprefix("posterior_"): archive[key]
                    for key in archive.files
                    if key.startswith("posterior_")
                }
                samples = flatten_samples(
                    saved, provenance["sampling"]["predictive_draws"]
                )
                result = predictive_components(
                    samples,
                    basis[target["_design_index"].to_numpy(dtype=int)],
                    degrees,
                    mean,
                    residual,
                    provenance["sampling"]["seed"]
                    + (0 if stage == "evaluation" else 100)
                    + 1,
                    provenance["sampling"]["degree_taper_width"],
                )
                if not np.allclose(result["full"], probabilities, atol=2e-6):
                    raise ValueError(
                        "No se pudo reproducir la predictiva guardada para medir sensibilidad"
                    )
                spatial_influence = total_variation(
                    result["full"], result["without_spatial"]
                )
                phylogenetic_influence = total_variation(
                    result["full"], result["without_phylogenetic"]
                )
            else:
                spatial_influence = archive["spatial_influence"]
                phylogenetic_influence = archive["phylogenetic_influence"]
            selected = rows.loc[ids]
            if (selected.role != role).any() or set(ids) != set(
                rows.index[rows.role.eq(role)]
            ):
                raise ValueError("IDs o roles incompatibles con las muestras")
            if not np.allclose(probabilities.sum(axis=-1), 1, atol=1e-6):
                raise ValueError("Probabilidades no normalizadas")
            if not np.allclose(
                selected[PROBABILITY_COLUMNS].astype(float),
                probabilities.mean(axis=0),
                atol=1e-6,
            ):
                raise ValueError("Medias del CSV no coinciden con las muestras")
            for index, (identifier, row) in enumerate(selected.iterrows()):
                cases.append(
                    {
                        "id": identifier,
                        "name": row.Name,
                        "family": row.familia_low or "Sin familia documentada",
                        "stage": stage,
                        "observed": row.observado,
                        "predicted": row.predicho,
                        "distributions": [
                            distribution_summary(
                                probabilities[:, index, c], np.linspace(0, 1, 21)
                            )
                            for c in range(3)
                        ],
                        "component_sensitivity": {
                            "spatial": distribution_summary(
                                spatial_influence[:, index], np.linspace(0, 1, 21)
                            ),
                            "phylogenetic": distribution_summary(
                                phylogenetic_influence[:, index], np.linspace(0, 1, 21)
                            ),
                        },
                    }
                )
            parameters = {}
            for name in [
                "max_degree",
                "sigma_spatial",
                "sigma_phylogenetic",
                "smoothing",
            ]:
                values = archive["posterior_" + name]
                parameters[name] = distribution_summary(values)
                # Las trazas son una vista resumida; los diagnósticos usan todos los draws.
                indices = np.linspace(
                    0, values.shape[1] - 1, min(200, values.shape[1]), dtype=int
                )
                parameters[name]["traces"] = values[:, indices].tolist()
                parameters[name]["draw_indices"] = (indices + 1).tolist()
            for c in range(2):
                parameters[f"intercept_{c + 1}"] = distribution_summary(
                    archive["posterior_intercept"][:, :, c]
                )
            stages[stage] = {
                "parameters": parameters,
                "degree_summary": provenance[diagnostic]["degree_summary"],
                "diagnostics": provenance[diagnostic],
                "n_fit": len(archive["fit_ids"]),
                "latent_shapes": {
                    name: list(archive["posterior_" + name].shape[2:])
                    for name in ["z_spatial", "z_phylogenetic"]
                    if "posterior_" + name in archive
                },
            }
    return {
        "cases": cases,
        "stages": stages,
        "classes": CLASSES,
        "status": provenance["status"],
        "sampling": provenance["sampling"],
    }


def build_posterior_explorer(folder, provenance, predictions):
    """Genera un HTML portable con JS/CSS locales incorporados y sin CDN."""
    payload = build_posterior_payload(folder, provenance, predictions)
    from .bayesian_context import build_bayesian_context

    payload["workflow"] = build_bayesian_context(provenance)
    assets = Path(__file__).parent / "posterior_assets"
    html = (assets / "index.html").read_text(encoding="utf-8")
    html = html.replace("__STYLE__", (assets / "style.css").read_text(encoding="utf-8"))
    html = html.replace(
        "__SCRIPT__",
        (assets / "app.js").read_text(encoding="utf-8")
        + "\n"
        + (assets / "workflow.js").read_text(encoding="utf-8"),
    )
    html = html.replace(
        "__DATA__",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
            "<", "\\u003c"
        ),
    )
    (folder / "posterior.html").write_text(html, encoding="utf-8")

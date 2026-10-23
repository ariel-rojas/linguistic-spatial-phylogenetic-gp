"""Informe de ejecuciones disponibles; declara pendientes sin inventar resultados."""

import json

import pandas as pd

from ..configuration import ROOT


def comparison_report():
    """Verifica población y test compartidos antes de comparar resultados guardados."""
    names = {
        "phylogenetic_mcc": "Matérn sobre cuerda",
        "geodesic_exponential": "Exponencial geodésico",
        "spherical_spline": "Spline esférico",
    }
    runs, test_ids, pending = {}, [], []
    for name in names:
        folder = ROOT / "results" / name
        if not (folder / "run.json").exists():
            pending.append(f"{names[name]}: búsqueda sin resultado completo.")
            continue
        runs[name] = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        rows = pd.read_csv(folder / "predictions.csv", keep_default_na=False)
        test_ids.append(set(rows.loc[rows.role.eq("test"), "ID"]))
    if not runs:
        raise ValueError("No hay búsquedas completas para comparar")
    if len({run["data_sha256"] for run in runs.values()}) != 1 or any(
        ids != test_ids[0] for ids in test_ids
    ):
        raise ValueError(
            "Los experimentos no comparten datos y test; no son comparables directamente"
        )
    tree_files = [run["phylogeny"]["files"] for run in runs.values()]
    if any(files != tree_files[0] for files in tree_files):
        raise ValueError("Los experimentos utilizan artefactos filogenéticos distintos")
    lines = [
        "# Comparación de componentes espaciales",
        "",
        "Búsquedas completas disponibles con la misma matriz MCC y el mismo test aleatorio estratificado por identidad.",
        "La selección utiliza solo CV interna de train. El test ya fue consultado en iteraciones anteriores: comparación exploratoria, no confirmación independiente.",
        "",
        "## Parámetros seleccionados dentro de train",
        "",
        "| Modelo | Candidatos | Log-loss CV | Accuracy test | Macro-F1 | Log-loss test | Recall Sin dominante |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in names.items():
        if name not in runs:
            continue
        run = runs[name]
        metrics, search = run["metrics"], run["search"]
        lines.append(
            f"| {label} | {search['n_candidates']} | {search['best_mean_validation_log_loss']:.4f} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | {metrics['log_loss']:.4f} | {metrics['per_class']['Sin dominante']['recall']:.4f} |"
        )
    lines += ["", "### Configuraciones ganadoras", ""]
    for name, label in names.items():
        if name not in runs:
            continue
        search = runs[name]["search"]
        lines += [
            f"- **{label}:** `{json.dumps(search['best_parameters'], ensure_ascii=False)}`. Extremos de grilla: {', '.join(search['boundary_parameters']) or 'ninguno'}."
        ]
    lines += [
        "",
        "## Comparación con parámetros fijos",
        "",
        "Los valores principales de los JSON mantienen a=1 y q=0.008. Los parámetros de forma son propios de cada estrategia; no son equivalencias de alcance.",
        "",
        "| Modelo | Protocolo | N test | Accuracy | Macro-F1 | Log-loss |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, label in names.items():
        folder = ROOT / "results" / name
        if not (folder / "fixed/run.json").exists():
            pending.append(f"{label}: referencia fija pendiente.")
            continue
        fixed = json.loads((folder / "fixed/run.json").read_text(encoding="utf-8"))
        reference = next(iter(runs.values()))
        if fixed["data_sha256"] != reference["data_sha256"]:
            raise ValueError("Referencia fija con datos diferentes")
        if fixed["phylogeny"]["files"] != reference["phylogeny"]["files"]:
            raise ValueError("Referencia fija con filogenia diferente")
        config = fixed["configuration"]
        if config["spatial_variance"] != 1 or config["phylogenetic_rate"] != 0.008:
            raise ValueError(
                "La comparación fija requiere a=1 y q=0.008 en los tres modelos"
            )
        protocols = {"Aleatorio": fixed["metrics"]}
        if "held_out_family_metrics" in fixed:
            protocols["Familia reservada (" + config["test_family"] + ")"] = fixed[
                "held_out_family_metrics"
            ]
        if (folder / "spatial/run.json").exists():
            spatial = json.loads((folder / "spatial/run.json").read_text(encoding="utf-8"))
            if (
                spatial["configuration"] != config
                or spatial["data_sha256"] != fixed["data_sha256"]
                or spatial["phylogeny"]["files"] != reference["phylogeny"]["files"]
            ):
                raise ValueError("Protocolo espacial desactualizado")
            protocols[
                f"Espacial ({spatial['block_km']:g} km; buffer {spatial['buffer_km']:g} km)"
            ] = spatial["pooled_by_language"]
        else:
            pending.append(f"{label}: evaluación espacial pendiente.")
        for protocol, metrics in protocols.items():
            lines.append(
                f"| {label} | {protocol} | {metrics['n_test']} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | {metrics['log_loss']:.4f} |"
            )
    lines += ["", "## Ejecuciones pendientes", ""]
    lines += [f"- {item}" for item in pending] or ["Ninguna en los protocolos anteriores."]
    lines += [
        "",
        "No se seleccionan parámetros espaciales o familiares con el ganador del split aleatorio. Las evaluaciones tienen objetivos distintos y no se comparan como si fueran la misma población de test.",
        "",
        "## Cartografía descriptiva",
        "",
    ]
    geography = ROOT / "datos/geography/manifest.json"
    if geography.exists():
        metadata = json.loads(geography.read_text(encoding="utf-8"))
        lines += [f"- {key}: {value}." for key, value in metadata["counts"].items()]
        lines += [
            f"- Ubicaciones dentro de regiones GMBA: {metadata['mountain_locations']}."
        ]
    lines += [
        "",
        "Natural Earth y GMBA solo aportan contexto. No se han impuesto barreras, cambiado coordenadas ni descartado registros por estas capas.",
        "",
        "Consulta cada `run.json` para matrices de confusión, clases, hashes y parámetros; `spatial/run.json` contiene folds y exclusiones por buffer. El mapa principal recomendado está en [spherical_spline/map.html](spherical_spline/map.html).",
        "",
    ]
    path = ROOT / "results/comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)

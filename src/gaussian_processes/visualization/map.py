"""Mapa local: color interior = orden; borde = resultado fuera de muestra."""

import hashlib
import json
from html import escape
from urllib.parse import quote

import folium
import pandas as pd
from folium.plugins import Fullscreen

from ..configuration import PROBABILITY_COLUMNS, ROOT, load_configuration
from ..data.geography import load_geography
from ..data.phylogeny import load_phylogeny
from ..models.spatial import spatial_kind
from .colors import COLORS, LABELS
from .geography import add_geographic_layers, geographic_details

BORDERS = {
    "Correcto": "#15803d",
    "Incorrecto": "#dc2626",
    "Entrenamiento": "#64748b",
    "Imputado sin verificar": "#475569",
    "Observado": "#334155",
    "Faltante sin imputar": "#334155",
}


def add_layer(map_, rows, title, show=False, overlay=False):
    """Añade puntos con color de orden y borde de estado, más su ficha de datos."""
    features = []
    for row in rows.fillna("").to_dict("records"):
        order = row["observado"]
        predicted = row.get("predicho", "")
        state = row.get("estado", "Observado")
        probabilities = [row.get(column, "") for column in PROBABILITY_COLUMNS]
        has_probabilities = all(value != "" for value in probabilities)
        probability = "Sin evaluación fuera de muestra"
        fill = COLORS[order]
        if has_probabilities:
            probability = "".join(
                f'<div style="margin-top:7px">{escape(LABELS[label])}: '
                f'<b>{float(value):.1%}</b><div style="background:#f1f5f9;height:7px;border-radius:4px">'
                f'<div style="background:{COLORS[label]};width:{float(value) * 100:.2f}%;height:7px;border-radius:4px"></div></div></div>'
                for label, value in zip(["OV", "Sin dominante", "VO"], probabilities)
            )
            fill = COLORS[predicted]
            if row.get("p_OV_lower", "") != "":
                probability += (
                    "<p><b>Intervalos creíbles del 95% de las probabilidades</b><br>"
                )
                probability += (
                    "<br>".join(
                        f"{escape(label)}: {float(row[column + '_lower']):.1%}–{float(row[column + '_upper']):.1%}"
                        for label, column in zip(
                            ["OV", "Sin dominante", "VO"], PROBABILITY_COLUMNS
                        )
                    )
                    + "</p>"
                )
        text = (
            '<div style="font:13px/1.5 system-ui;min-width:240px;padding:4px">'
            f"<b style='font-size:16px'>{escape(row['Name'])}</b><br>"
            f"WALS: {escape(row['ID'])} · Glottocode: {escape(row['Glottocode'])}<br>"
            f"Familia registrada: {escape(row.get('familia_low', '')) or 'No disponible'}<hr>"
            f"<b>Observado:</b> {escape(LABELS[order])}<br>"
            f"<b>Predicción:</b> {escape(LABELS.get(predicted, 'No evaluada'))}<br>"
            f"<b>Estado:</b> {escape(state)}<br>{probability}<hr>"
            + (
                f'<p><a href="posterior.html?language={quote(row["ID"], safe="")}&amp;stage={row["_posterior_stage"]}" '
                'target="_blank" rel="noopener"><b>Explorar distribuciones e incertidumbre →</b></a></p>'
                if row.get("_posterior_stage")
                else ""
            )
            + f"{geographic_details(row)}"
            f"{('Sin imputación: ' + escape(row.get('motivo_exclusion', ''))) if state == 'Faltante sin imputar' else ''}"
            f"<a href='https://wals.info/languoid/lect/wals_code_{escape(row['ID'])}' target='_blank'>Consultar WALS</a></div>"
        )
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row["Longitude"]), float(row["Latitude"])],
                },
                "properties": {
                    "nombre": row["Name"],
                    "resumen": (
                        f"Predicción: {LABELS.get(predicted, predicted)} · "
                        f"P máx.: {max(float(v) for v in probabilities):.0%}"
                    )
                    if has_probabilities
                    else LABELS[order],
                    "detalle": text,
                    "fill": fill,
                    "border": BORDERS[state],
                    "radius": 6 if state in ("Correcto", "Incorrecto") else 4,
                    "weight": 2 if state in ("Correcto", "Incorrecto") else 1,
                },
            }
        )
    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name=title,
        overlay=overlay,
        show=show,
        marker=folium.CircleMarker(fill=True, fill_opacity=1),
        style_function=lambda f: {
            "fillColor": f["properties"]["fill"],
            "color": f["properties"]["border"],
            "radius": f["properties"]["radius"],
            "weight": f["properties"]["weight"],
        },
        tooltip=folium.GeoJsonTooltip(fields=["nombre", "resumen"], labels=False),
        popup=folium.GeoJsonPopup(fields=["detalle"], labels=False, max_width=380),
    ).add_to(map_)


def imputation_view(observations, imputations):
    """Integra predicciones sin alterar la columna de observaciones originales.

    El enlace uno a uno por ID evita dibujar dos puntos para una misma fila.
    Los faltantes excluidos conservan su motivo y no reciben etiquetas inventadas.
    """
    columns = ["ID", "predicho", *PROBABILITY_COLUMNS]
    if "_posterior_stage" in imputations:
        columns.append("_posterior_stage")
    columns += [
        c + suffix
        for c in PROBABILITY_COLUMNS
        for suffix in ["_lower", "_upper"]
        if c + suffix in imputations
    ]
    view = observations.merge(
        imputations[columns], on="ID", how="left", validate="one_to_one"
    )
    view["estado"] = "Observado"
    view.loc[view.observado.eq("Faltante"), "estado"] = "Faltante sin imputar"
    view.loc[view.predicho.notna(), "estado"] = "Imputado sin verificar"
    return view


def build_map(filename, fixed=False, bayesian=False):
    """Genera observaciones, evaluación aleatoria e imputación multiclase.

    Las imputaciones se integran por ID en su vista, sin superponer el
    punto gris original. Los excluidos siguen grises con motivo explícito.
    """
    config, folder = load_configuration(filename)
    if fixed and bayesian:
        raise ValueError("Elige --fixed o --bayesian, no ambos")
    if bayesian:
        folder = folder / "bayesian"
    if fixed:
        folder = folder / "fixed"
    provenance = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    if bayesian and provenance.get("status") not in {"complete", "diagnostics_warning"}:
        raise ValueError(
            "El ajuste bayesiano aún no terminó; no se muestran predicciones anteriores"
        )
    saved = provenance["configuration"]
    if saved != config:
        raise ValueError(
            "La configuración cambió. Ejecuta gaussian-processes search antes de generar el mapa"
        )
    if (
        provenance["data_sha256"]
        != hashlib.sha256((ROOT / "datos/lenguas.csv").read_bytes()).hexdigest()
    ):
        raise ValueError(
            "Los datos cambiaron: vuelve a ejecutar el experimento antes del mapa"
        )
    all_rows = pd.read_csv(ROOT / "datos/lenguas.csv", keep_default_na=False)
    _, current_phylogeny = load_phylogeny(config, all_rows)
    if provenance.get("phylogeny", {}).get("files") != current_phylogeny["files"]:
        raise ValueError(
            "El árbol o su matriz cambiaron; vuelve a ejecutar el experimento"
        )
    saved_covariance = provenance.get("phylogeny", {}).get("model_covariance")
    if (
        saved_covariance is not None
        and saved_covariance != current_phylogeny["model_covariance"]
    ):
        raise ValueError(
            "Cambió la covarianza aplicada; vuelve a ejecutar el experimento"
        )
    predictions = pd.read_csv(folder / "predictions.csv", keep_default_na=False)
    if bayesian:
        from .posterior import build_posterior_explorer

        build_posterior_explorer(folder, provenance, predictions)
        predictions["_posterior_stage"] = predictions.role.map(
            {"test": "evaluation", "imputed": "imputation"}
        ).fillna("")
    geography, geography_manifest = load_geography()
    if geography is not None:
        all_rows = all_rows.merge(geography, on="ID", how="left", validate="one_to_one")
        predictions = predictions.merge(
            geography, on="ID", how="left", validate="one_to_one"
        )
    evaluation = predictions.loc[predictions.role.isin(["train", "test"])]
    imputed = predictions.loc[predictions.role.eq("imputed")]
    map_ = folium.Map(
        location=[10, 5], zoom_start=1, tiles=None, prefer_canvas=True, min_zoom=1
    )
    folium.TileLayer("OpenStreetMap", overlay=True, control=False).add_to(map_)
    if geography_manifest is not None:
        add_geographic_layers(map_)
    add_layer(
        map_,
        all_rows.loc[all_rows.coordenadas_validas],
        "1 · Datos observados en WALS",
    )
    counts = evaluation.estado.value_counts()
    title = f"2 · Test: {counts.get('Correcto', 0)} aciertos · {counts.get('Incorrecto', 0)} errores"
    add_layer(
        map_, evaluation.loc[evaluation.estado.ne("Entrenamiento")], title, show=True
    )
    training = evaluation.loc[evaluation.estado.eq("Entrenamiento")]
    add_layer(
        map_, training, f"3 · Train: {len(training)} observaciones (sin evaluación)"
    )
    add_layer(map_, evaluation, f"4 · Train + test: {len(evaluation)} registros")
    if fixed and config["test_family"] is not None:
        family_rows = pd.read_csv(
            folder / "family_evaluation.csv", keep_default_na=False
        )
        if geography is not None:
            family_rows = family_rows.merge(
                geography, on="ID", how="left", validate="one_to_one"
            )
        family_counts = family_rows.estado.value_counts()
        title = (
            f"Familia reservada: {config['test_family']} — "
            f"{family_counts.get('Correcto', 0)} aciertos, {family_counts.get('Incorrecto', 0)} errores"
        )
        add_layer(map_, family_rows.loc[family_rows.estado.ne("Entrenamiento")], title)
    completed = imputation_view(all_rows, imputed)
    no_prediction = completed.estado.eq("Faltante sin imputar").sum()
    add_layer(
        map_,
        completed.loc[completed.coordenadas_validas],
        f"5 · Imputaciones: {len(imputed)} propuestas · {no_prediction} pendientes¹",
    )
    Fullscreen().add_to(map_)
    folium.LayerControl(collapsed=True).add_to(map_)
    swatches = "".join(
        f'<div style="margin:7px 0"><span style="color:{COLORS[label]}">●</span> <b>{LABELS[label]}</b></div>'
        for label in ["OV", "VO", "Sin dominante"]
    )
    model_label = {
        "spherical_spline": "Spline esférico + MCC",
        "geodesic_exponential": "Exponencial geodésico + MCC",
        "matern_chord": "Matérn sobre cuerda + MCC",
    }[spatial_kind(config)]
    fit_label = "Parámetros fijos" if fixed else "Parámetros seleccionados en train"
    if bayesian:
        fit_label = "NUTS · intervalos de probabilidad del 95%"
        if provenance.get("status") != "complete":
            fit_label += " · Diagnósticos pendientes: resultados exploratorios"
        fit_label += '<br><a href="posterior.html" target="_blank" rel="noopener">Explorar distribuciones por lengua →</a>'
    map_.get_root().html.add_child(
        folium.Element(
            "<style>.leaflet-tile-pane{filter:saturate(0.35)}.leaflet-control-layers{font:13px/1.6 system-ui;max-width:360px}"
            ".leaflet-popup-content{margin:12px}.leaflet-popup-content hr{margin:8px 0;border:0;border-top:1px solid #e2e8f0}.leaflet-tooltip{font:13px system-ui}</style>"
            '<div style="position:fixed;bottom:26px;left:12px;z-index:9999;background:white;'
            "padding:14px;border-radius:10px;box-shadow:0 2px 12px #0003;"
            'font:13px/1.5 system-ui;width:270px;max-height:65vh;overflow:auto">'
            '<b style="font-size:16px">Orden de objeto y verbo</b><br>'
            f"<small>{model_label} · {fit_label}</small><br>"
            "Color pleno = clase · probabilidades en la ficha" + swatches + ""
            '<hr><span style="color:#15803d">●</span> Borde verde: acierto en test<br>'
            '<span style="color:#dc2626">●</span> Borde rojo: error en test<br>'
            "Gris: dato faltante sin imputación<br>"
            '<details style="margin-top:8px"><summary>Cómo leer las capas</summary>'
            "1: colores de observaciones, sin degradé.<br>"
            "2: predicciones solo sobre el test reservado.<br>"
            "3: train observado, borde gris; no cuenta como acierto.<br>"
            "4: train observado + test predicho.<br>"
            "5: observaciones + propuestas para faltantes.<br>"
            "Las propuestas tienen borde oscuro y no están verificadas. "
            "Pulsa una lengua para ver las tres probabilidades.<br>"
            "¹ Pendientes incluye registros sin coordenadas, invisibles en el mapa.<hr>"
            "<small>WALS 83A: Dryer (2013), CC BY 4.0.<br>"
            "Filogenia: Bouckaert et al., MCC v2.0.0.<br>"
            "Sin dominante es una clase observada, no incertidumbre. "
            "La probabilidad máxima no garantiza calibración. Requiere internet.</small>"
            "</details></div>"
        )
    )
    map_.save(folder / "map.html")
    print(folder / "map.html")

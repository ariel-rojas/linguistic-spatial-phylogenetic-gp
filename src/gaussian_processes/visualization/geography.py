"""Capas de contexto físico y ficha descriptiva; no representan barreras inferidas."""

import json
from html import escape

import folium

from ..data.geography import FOLDER


def geographic_details(row):
    """Texto seguro para el popup, sin equiparar ausencia de polígono con llanura."""
    if "geographic_status" not in row:
        return ""
    return (
        "<hr><b>Contexto físico de la ubicación</b><br>"
        + escape(row["geographic_status"])
        + "<br>Región GMBA: "
        + escape(row.get("mountain_region", "") or "Fuera de polígonos / no asignada")
        + "<br><small>Metadatos actuales; no modifican la predicción.</small>"
    )


def add_geographic_layers(map_):
    """Añade dos overlays opcionales simplificados, conservando el mapa de puntos."""
    context = json.loads((FOLDER / "context.geojson").read_text(encoding="utf-8"))
    pane = folium.map.CustomPane("physical-context", z_index=350, pointer_events=True)
    pane.add_to(map_)
    for kind, title, color in (
        ("land", "Contexto · Tierra y costas (Natural Earth)", "#475569"),
        ("mountains", "Contexto · Regiones montañosas (GMBA)", "#927341"),
    ):
        collection = {
            "type": "FeatureCollection",
            "features": [
                feature
                for feature in context["features"]
                if feature["properties"]["layer"] == kind
            ],
        }
        folium.GeoJson(
            collection,
            name=title,
            show=False,
            overlay=True,
            pane="physical-context",
            style_function=lambda feature, color=color: {
                "color": color,
                "weight": 0.7,
                "fillColor": color,
                "fillOpacity": 0.12,
            },
            tooltip=folium.GeoJsonTooltip(fields=["name"], labels=False),
        ).add_to(map_)
    map_.get_root().html.add_child(
        folium.Element(
            '<div style="position:fixed;bottom:0;left:0;background:white;z-index:1000;font:10px system-ui">'
            "Contexto: Natural Earth (dominio público); "
            '<a href="https://doi.org/10.1038/s41597-022-01256-y">Snethlage et al. (2022)</a>, '
            '<a href="https://doi.org/10.48601/earthenv-t9k2-1407">GMBA v2 (CC BY 4.0)</a>. '
            "Polígonos de visualización simplificados.</div>"
        )
    )

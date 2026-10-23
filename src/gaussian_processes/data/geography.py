"""Cartografía física descriptiva; nunca modifica etiquetas ni elegibilidad.

Natural Earth 5.1.2: dominio público. GMBA v2 Standard Basic: CC BY 4.0.
Los polígonos originales clasifican puntos; la copia simplificada solo se dibuja.
Las coordenadas WALS representan ubicaciones, no toda el área de una lengua.
"""

import hashlib
import json
import shutil
from datetime import UTC, datetime
from urllib.request import Request, urlopen

import geopandas as gpd
import pandas as pd

from ..configuration import ROOT
from .loading import load_data

FOLDER = ROOT / "datos/geography"
SOURCES = {
    "land": {
        "url": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/ne_10m_land.geojson",
        "filename": "land.geojson",
        "version": "Natural Earth 5.1.2, 1:10 millones",
        "license": "Public domain",
    },
    "ocean": {
        "url": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/ne_10m_ocean.geojson",
        "filename": "ocean.geojson",
        "version": "Natural Earth 5.1.2, 1:10 millones",
        "license": "Public domain",
    },
    "mountains": {
        "url": "https://data.earthenv.org/mountains/standard/GMBA_Inventory_v2.0_standard_basic.zip",
        "filename": "mountains.zip",
        "version": "GMBA v2 Standard Basic",
        "license": "CC BY 4.0",
        "publication": "https://doi.org/10.1038/s41597-022-01256-y",
        "dataset": "https://doi.org/10.48601/earthenv-t9k2-1407",
    },
}


def file_hash(path):
    """Hash por bloques: permite auditar las fuentes sin cargarlas completas."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_source(source, destination):
    """Descarga atómica; reutiliza fuentes locales para permitir trabajo sin red."""
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        request = Request(
            source["url"],
            headers={"User-Agent": "Mozilla/5.0 (compatible; LinguisticResearch/1.0)"},
        )
        with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)


def classify_locations(data, land, ocean, mountains):
    """Cruza puntos válidos; guarda todos los registros, también los no ubicables.

    Un punto fuera de tierra no se corrige ni se descarta. Los límites o huecos
    cartográficos se marcan para revisión. Intersecciones múltiples se conservan.
    """
    valid = data.loc[data.coordenadas_validas].copy()
    points = gpd.GeoDataFrame(
        valid[["ID"]],
        geometry=gpd.points_from_xy(valid.Longitude, valid.Latitude),
        crs=4326,
    ).set_index("ID")

    def memberships(polygons):
        joined = gpd.sjoin(points, polygons, how="inner", predicate="intersects")
        return joined

    on_land = set(memberships(land[["geometry"]]).index)
    on_ocean = set(memberships(ocean[["geometry"]]).index)
    mountain_matches = memberships(mountains[["mountain_name", "geometry"]])
    mountain_names = mountain_matches.groupby(level=0).mountain_name.agg(
        lambda values: "; ".join(sorted(set(values.astype(str))))
    )
    rows = data[["ID"]].copy()
    rows["geographic_status"] = "Sin coordenadas válidas"
    rows.loc[rows.ID.isin(points.index), "geographic_status"] = (
        "Sin cobertura terrestre/marina: revisar"
    )
    rows.loc[rows.ID.isin(on_ocean), "geographic_status"] = (
        "Océano cartografiado: revisar ubicación"
    )
    rows.loc[rows.ID.isin(on_land), "geographic_status"] = "Tierra cartografiada"
    rows.loc[rows.ID.isin(on_land & on_ocean), "geographic_status"] = (
        "Límite costero: revisar"
    )
    rows["mountain_region"] = rows.ID.map(mountain_names).fillna("")
    return rows


def prepare_geography():
    """Prepara una tabla de atributos, una capa ligera y un manifiesto trazable."""
    FOLDER.mkdir(parents=True, exist_ok=True)
    source_folder = FOLDER / "sources"
    source_folder.mkdir(exist_ok=True)
    layers, provenance = {}, {}
    for name, source in SOURCES.items():
        path = source_folder / source["filename"]
        print(f"Cartografía: {source['version']} ({name})", flush=True)
        fetch_source(source, path)
        layers[name] = gpd.read_file(path).to_crs(4326)
        layers[name].geometry = layers[name].geometry.make_valid()
        provenance[name] = {**source, "sha256": file_hash(path)}
    mountains = layers["mountains"]
    # El esquema GMBA se valida explícitamente; no adivinamos otra columna.
    if "MapName" not in mountains:
        raise ValueError(
            f"Falta MapName en GMBA; columnas disponibles: {list(mountains.columns)}"
        )
    mountains["mountain_name"] = mountains["MapName"].fillna("Región sin nombre")
    rows = classify_locations(load_data(), layers["land"], layers["ocean"], mountains)
    rows.to_csv(FOLDER / "languages.csv", index=False)
    display = []
    for name in ("land", "mountains"):
        layer = layers[name][["geometry"]].copy()
        layer["layer"] = name
        layer["name"] = (
            mountains.mountain_name
            if name == "mountains"
            else "Tierra y costa — Natural Earth"
        )
        # Tolerancia angular únicamente para dibujo; jamás para asignar atributos.
        layer.geometry = layer.geometry.simplify(0.03, preserve_topology=True)
        display.append(layer)
    context = gpd.GeoDataFrame(pd.concat(display, ignore_index=True), crs=4326)
    (FOLDER / "context.geojson").write_text(
        context.to_json(drop_id=True), encoding="utf-8"
    )
    manifest = {
        "prepared_at": datetime.now(UTC).isoformat(),
        "data_sha256": file_hash(ROOT / "datos/lenguas.csv"),
        "sources": provenance,
        "files": {
            name: file_hash(FOLDER / name)
            for name in ("languages.csv", "context.geojson")
        },
        "counts": rows.geographic_status.value_counts().to_dict(),
        "mountain_locations": int(rows.mountain_region.ne("").sum()),
        "display_simplification_degrees": 0.03,
        "use": "Metadatos descriptivos: no intervienen en el kernel ni alteran coordenadas",
    }
    (FOLDER / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False), flush=True)


def load_geography():
    """Devuelve metadatos opcionales, verificando datos y archivos derivados."""
    path = FOLDER / "manifest.json"
    if not path.exists():
        return None, None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["data_sha256"] != file_hash(ROOT / "datos/lenguas.csv"):
        raise ValueError("Cartografía desactualizada: ejecuta prepare-geography")
    for name, expected in manifest["files"].items():
        if file_hash(FOLDER / name) != expected:
            raise ValueError(
                f"Cartografía alterada ({name}): ejecuta prepare-geography"
            )
    return pd.read_csv(FOLDER / "languages.csv", keep_default_na=False), manifest

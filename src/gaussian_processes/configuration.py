"""Configuración validada; las rutas relativas parten del repositorio."""

import json
import math
import re
from pathlib import Path

from .models.spatial import SPATIAL_PARAMETERS, spatial_kind

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["OV", "Sin dominante", "VO"]  # Orden alfabético de sklearn.
PROBABILITY_COLUMNS = ["p_OV", "p_SD", "p_VO"]
DEFAULT_CONFIGURATION = "config/spherical_spline_family.json"


def validate_positive(value, name):
    """Rechaza booleanos, infinitos y valores no positivos."""
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} debe ser un número finito positivo")


def load_configuration(filename=DEFAULT_CONFIGURATION):
    """Valida parámetros y grilla; reutilizar name reemplaza sus resultados."""
    path = Path(filename)
    if not path.is_absolute():
        path = ROOT / path
    config = json.loads(path.read_text(encoding="utf-8"))
    kind = spatial_kind(config)
    if kind not in SPATIAL_PARAMETERS:
        raise ValueError(f"spatial_kernel desconocido: {kind}")
    parameters = [*SPATIAL_PARAMETERS[kind], "spatial_variance", "phylogenetic_rate"]
    expected = {
        "name",
        "test_fraction",
        "phylogeny_manifest",
        "seed",
        "test_family",
        "cv_folds",
        "grid",
        *parameters,
    }
    if "spatial_kernel" in config:
        expected.add("spatial_kernel")
    for key in ("phylogenetic_structure", "phylogenetic_normalize"):
        if key in config:
            expected.add(key)
    if config.get("phylogenetic_structure", "global") not in {"global", "within_family"}:
        raise ValueError("phylogenetic_structure debe ser global o within_family")
    if type(config.get("phylogenetic_normalize", False)) is not bool:
        raise ValueError("phylogenetic_normalize debe ser booleano")
    if set(config) != expected:
        raise ValueError(f"Claves requeridas: {sorted(expected)}")
    if not isinstance(config["name"], str) or not re.fullmatch(
        r"[a-zA-Z0-9_-]+", config["name"]
    ):
        raise ValueError("name debe contener letras, números, guiones o guiones bajos")
    if (
        not isinstance(config["phylogeny_manifest"], str)
        or not config["phylogeny_manifest"].strip()
    ):
        raise ValueError("phylogeny_manifest debe ser una ruta")
    for key in parameters:
        validate_positive(config[key], key)
    if kind == "spherical_spline" and type(config["max_degree"]) is not int:
        raise ValueError("max_degree debe ser entero")
    validate_positive(config["test_fraction"], "test_fraction")
    if config["test_fraction"] >= 1:
        raise ValueError("test_fraction debe ser menor que 1")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed debe ser un entero entre 0 y 2**32 - 1")
    if type(config["cv_folds"]) is not int or config["cv_folds"] < 2:
        raise ValueError("cv_folds debe ser un entero de al menos 2")
    family = config["test_family"]
    if family is not None and (not isinstance(family, str) or not family.strip()):
        raise ValueError("test_family debe ser un identificador o null")
    if set(config["grid"]) != set(parameters):
        raise ValueError(f"La grilla debe definir {parameters}")
    for key, values in config["grid"].items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"grid.{key} debe ser una lista no vacía")
        for value in values:
            validate_positive(value, key)
            if key == "max_degree" and type(value) is not int:
                raise ValueError("grid.max_degree debe contener enteros")
        if len(set(values)) != len(values):
            raise ValueError(f"grid.{key} contiene valores repetidos")
    return config, ROOT / "results" / config["name"]

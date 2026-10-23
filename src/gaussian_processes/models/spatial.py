"""Estrategias espaciales intercambiables y sus parámetros de búsqueda."""

from dataclasses import dataclass

from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from .base import ColumnKernel, KernelComponent
from .spherical import GeodesicExponentialKernel, SphericalSplineKernel

# Nombres públicos y rutas de parámetros del kernel interno (sin amplitud).
SPATIAL_PARAMETERS = {
    "matern_chord": {"length_scale_km": "length_scale", "nu": "nu"},
    "geodesic_exponential": {"length_scale_km": "length_scale"},
    "spherical_spline": {"max_degree": "max_degree", "smoothing": "smoothing"},
}


def spatial_kind(config):
    """Las configuraciones históricas sin selector conservan Matérn sobre cuerda."""
    return config.get("spatial_kernel", "matern_chord")


def parameter_paths(config):
    """Rutas sklearn de la estrategia elegida; no busca parámetros irrelevantes."""
    return {
        **{
            key: f"kernel__k1__k2__kernel__{value}"
            for key, value in SPATIAL_PARAMETERS[spatial_kind(config)].items()
        },
        "spatial_variance": "kernel__k1__k1__constant_value",
        "phylogenetic_rate": "kernel__k2__k1__constant_value",
    }


@dataclass
class GeodesicExponentialComponent(KernelComponent):
    """Estrategia superficial con alcance en km y amplitud independiente."""

    length_scale_km: float
    variance: float

    def build(self):
        """Aplica amplitud al exponencial superficial usando únicamente xyz."""
        return ConstantKernel(self.variance, "fixed") * ColumnKernel(
            GeodesicExponentialKernel(self.length_scale_km), 0, 3
        )


@dataclass
class SphericalSplineComponent(KernelComponent):
    """Estrategia tipo GAM: base esférica regularizada y prior gaussiano propio."""

    max_degree: int
    smoothing: float
    variance: float

    def build(self):
        """Compone amplitud y suavizado esférico sin incluir el índice filogenético."""
        return ConstantKernel(self.variance, "fixed") * ColumnKernel(
            SphericalSplineKernel(self.max_degree, self.smoothing), 0, 3
        )


def build_spatial_component(config):
    """Selecciona una clase concreta sin modificar el clasificador ni la filogenia."""
    kind = spatial_kind(config)
    if kind == "spherical_spline":
        return SphericalSplineComponent(
            config["max_degree"], config["smoothing"], config["spatial_variance"]
        )
    if kind == "geodesic_exponential":
        return GeodesicExponentialComponent(
            config["length_scale_km"], config["spatial_variance"]
        )
    if kind == "matern_chord":
        return SpatialMaternComponent(
            config["length_scale_km"], config["nu"], config["spatial_variance"]
        )
    raise ValueError(f"Kernel espacial desconocido: {kind}")


@dataclass
class SpatialMaternComponent(KernelComponent):
    """Configura suavidad, alcance y amplitud; nu=0.5 es exponencial."""

    length_scale_km: float
    nu: float
    variance: float

    def build(self):
        """Solo las primeras tres columnas (xyz) participan en este componente."""
        return ConstantKernel(self.variance, "fixed") * ColumnKernel(
            Matern(self.length_scale_km, "fixed", self.nu), start=0, stop=3
        )

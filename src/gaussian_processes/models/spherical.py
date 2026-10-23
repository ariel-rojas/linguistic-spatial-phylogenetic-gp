"""Geometría esférica y kernels compatibles con sklearn, sin ajuste interno.

Las entradas son xyz; se normalizan para trabajar sobre la esfera unitaria.
Los parámetros se seleccionan mediante la grilla, no mediante gradientes.
"""

from abc import abstractmethod

import numpy as np
from numpy.polynomial.legendre import legval
from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import Kernel

EARTH_RADIUS_KM = 6371.0


def unit_vectors(X):
    """Valida puntos xyz finitos, no nulos, y devuelve direcciones unitarias."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3 or not np.isfinite(X).all():
        raise ValueError("Se requieren coordenadas xyz finitas de tres columnas")
    norms = np.linalg.norm(X, axis=1)
    if np.any(norms == 0):
        raise ValueError("El centro de la esfera no es una ubicación válida")
    return X / norms[:, None]


def great_circle_distance(X, Y=None):
    """Arcos en km; atan2 conserva precisión cerca de coincidencias y antípodas."""
    left = unit_vectors(X)
    right = left if Y is None else unit_vectors(Y)
    difference = cdist(left, right)
    opposite = cdist(left, -right)
    return EARTH_RADIUS_KM * 2 * np.arctan2(difference, opposite)


class SphericalKernel(Kernel):
    """Contrato común de correlaciones con diagonal uno y parámetros fijos.

    Son isotrópicos sobre la esfera, pero no estacionarios frente a traslaciones
    arbitrarias en el espacio xyz; por eso is_stationary devuelve False.
    """

    @abstractmethod
    def correlation(self, X, Y=None):
        """Calcula la correlación de la estrategia concreta sobre puntos xyz."""

    def __call__(self, X, Y=None, eval_gradient=False):
        """Devuelve bloques train/train o test/train y gradiente vacío."""
        if eval_gradient and Y is not None:
            raise ValueError("El gradiente solo se calcula con Y=None")
        result = self.correlation(X, Y)
        if Y is None:
            np.fill_diagonal(result, 1.0)
        if eval_gradient:
            return result, np.empty((*result.shape, 0))
        return result

    def diag(self, X):
        """Diagonal normalizada; no construye la matriz completa."""
        return np.ones(len(unit_vectors(X)))

    def is_stationary(self):
        """Isotropía esférica no implica invariancia ante traslaciones xyz."""
        return False


class GeodesicExponentialKernel(SphericalKernel):
    """Exponencial válido sobre arcos esféricos: exp(-d / length_scale).

    Equivale a Matérn nu=1/2 sobre distancia superficial. No admite otros nu.
    """

    def __init__(self, length_scale=1500.0):
        self.length_scale = length_scale

    def correlation(self, X, Y=None):
        """Convierte distancia superficial en correlación."""
        if not np.isfinite(self.length_scale) or self.length_scale <= 0:
            raise ValueError("length_scale debe ser positivo y finito")
        return np.exp(-great_circle_distance(X, Y) / self.length_scale)


class SphericalSplineKernel(SphericalKernel):
    """Suavizado espectral de rango finito con penalización cuadrática.

    Cada armónico de grado l tiene varianza 1/(1+smoothing*[l(l+1)]²).
    Esto corresponde a penalizar norma cuadrática + rugosidad (Laplaciano²).
    El término de norma hace propio el prior, incluida la constante l=0.
    Se incluyen todos los órdenes de cada grado para preservar isotropía.
    El teorema de adición permite evaluar Legendre sin materializar las bases.
    Se normaliza solo esta correlación espacial, nunca la matriz filogenética.
    No pretende reproducir exactamente el spline por defecto de mgcv.
    """

    def __init__(self, max_degree=16, smoothing=0.001):
        self.max_degree = max_degree
        self.smoothing = smoothing

    def correlation(self, X, Y=None):
        """Suma de kernels por grado con pesos positivos: covarianza válida."""
        if not isinstance(self.max_degree, (int, np.integer)) or self.max_degree < 1:
            raise ValueError("max_degree debe ser un entero positivo")
        if not np.isfinite(self.smoothing) or self.smoothing <= 0:
            raise ValueError("smoothing debe ser positivo y finito")
        left = unit_vectors(X)
        right = left if Y is None else unit_vectors(Y)
        cosine = np.clip(left @ right.T, -1, 1)
        degree = np.arange(self.max_degree + 1, dtype=float)
        weights = (2 * degree + 1) / (1 + self.smoothing * (degree * (degree + 1)) ** 2)
        return legval(cosine, weights / weights.sum())

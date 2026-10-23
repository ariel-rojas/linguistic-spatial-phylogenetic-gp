"""Adaptador de covarianza filogenética ya preparada: cruda o normalizada."""

from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel

from .base import ColumnKernel, KernelComponent


class PhylogeneticKernel(Kernel):
    """Adaptador sklearn de una matriz fija, indexada por hojas del árbol.

    X contiene una sola columna de índices globales. No son distancias ni
    variables ordinales. El constructor conserva covariance para permitir clone.
    La amplitud se configura fuera, con ConstantKernel.
    """

    def __init__(self, covariance):
        self.covariance = covariance

    def _indices(self, X):
        """Impide índices fraccionarios, negativos, ausentes o fuera de rango."""
        values = np.asarray(X)
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError("El kernel filogenético requiere una columna de índices")
        values = values[:, 0]
        if (
            not np.isfinite(values).all()
            or np.any(values != np.floor(values))
            or np.any(values < 0)
            or np.any(values >= len(self.covariance))
        ):
            raise ValueError("Índice de hoja inválido")
        return values.astype(np.intp)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Devuelve bloques rectangulares y un gradiente vacío (matriz fija)."""
        if eval_gradient and Y is not None:
            raise ValueError("El gradiente solo se calcula con Y=None")
        left = self._indices(X)
        right = left if Y is None else self._indices(Y)
        result = self.covariance[np.ix_(left, right)]
        if eval_gradient:
            return result, np.empty((*result.shape, 0))
        return result

    def diag(self, X):
        """Varianzas: duraciones raíz-hoja, sin construir una matriz completa."""
        indices = self._indices(X)
        return self.covariance[indices, indices]

    def is_stationary(self):
        """La identidad de una hoja no define una distancia euclídea."""
        return False

    def __repr__(self):
        """Representación breve: evita imprimir miles de valores de la matriz."""
        return f"PhylogeneticKernel(n_tips={len(self.covariance)})"


@dataclass
class BrownianPhylogeneticComponent(KernelComponent):
    """Multiplica la matriz preparada por rate.

    rate es varianza marginal con matriz normalizada; con C cruda representa
    varianza por milenio. Se conserva el nombre del parámetro por compatibilidad.
    """

    covariance: np.ndarray
    rate: float

    def build(self):
        """Selecciona la cuarta columna y multiplica C por la tasa evolutiva."""
        return ConstantKernel(self.rate, "fixed") * ColumnKernel(
            PhylogeneticKernel(self.covariance), start=3, stop=4
        )

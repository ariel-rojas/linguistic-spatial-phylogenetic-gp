"""Transformaciones de la covarianza fuente, sin alterar el árbol ni sus archivos."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhylogeneticCovariance:
    """Política de estructura y escala anterior al adaptador de sklearn.

    La normalización produce correlaciones Cij/sqrt(Cii*Cjj). El bloqueo
    conserva submatrices principales completas, por lo que preserva la
    semidefinición positiva. Las etiquetas familiares deben estar alineadas
    con las hojas; cada familia desconocida necesita una etiqueta individual.
    """

    structure: str = "global"
    normalize: bool = False

    def transform(self, covariance, families=None):
        """Devuelve una copia; nunca invierte ni modifica la matriz fuente."""
        if self.structure not in {"global", "within_family"}:
            raise ValueError("phylogenetic_structure debe ser global o within_family")
        if type(self.normalize) is not bool:
            raise ValueError("phylogenetic_normalize debe ser booleano")
        result = np.array(covariance, dtype=float, copy=True)
        if (
            result.ndim != 2
            or result.shape[0] != result.shape[1]
            or not np.isfinite(result).all()
            or not np.allclose(result, result.T)
            or np.any(result.diagonal() <= 0)
        ):
            raise ValueError("Se requiere covarianza simétrica finita con diagonal positiva")
        if self.normalize:
            scale = np.sqrt(result.diagonal())
            result /= scale[:, None] * scale[None, :]
            np.fill_diagonal(result, 1.0)
        if self.structure == "within_family":
            groups = np.asarray(families, dtype=str)
            if groups.shape != (len(result),) or np.any(groups == ""):
                raise ValueError("Faltan familias alineadas o bloques individuales")
            result *= groups[:, None] == groups[None, :]
        return result

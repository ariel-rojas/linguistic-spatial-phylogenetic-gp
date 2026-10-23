"""Álgebra determinista del GP, independiente del motor de inferencia."""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import sph_harm_y

# Dos contrastes ortogonales y simétricos: cada logit tiene varianza unitaria
# si los dos contrastes tienen varianza uno. El orden sigue CLASSES.
CLASS_CONTRASTS = np.array(
    [[1, 1 / np.sqrt(3)], [-1, 1 / np.sqrt(3)], [0, -2 / np.sqrt(3)]]
) * np.sqrt(3 / 4)


def spherical_basis(latitude, longitude, max_degree):
    """Armónicos reales completos; su producto ponderado reproduce el spline.

    Incluye grado cero para conservar exactamente el kernel existente. El
    intercepto adicional tiene prior propio; no interpretamos por separado
    ese intercepto y el modo espacial constante.
    """
    theta = np.pi / 2 - np.radians(np.asarray(latitude))
    phi = np.radians(np.asarray(longitude))
    columns, degrees = [], []
    for degree in range(max_degree + 1):
        columns.append(sph_harm_y(degree, 0, theta, phi).real)
        degrees.append(degree)
        for order in range(1, degree + 1):
            harmonic = sph_harm_y(degree, order, theta, phi)
            columns.extend([np.sqrt(2) * harmonic.real, np.sqrt(2) * harmonic.imag])
            degrees.extend([degree, degree])
    return np.column_stack(columns) * np.sqrt(4 * np.pi), np.asarray(degrees)


def basis_scales(degrees, smoothing):
    """Desviación de coeficientes que da diagonal uno al kernel espacial."""
    weights = 1 / (1 + smoothing * (degrees * (degrees + 1)) ** 2)
    return np.sqrt(weights / weights.sum())


def covariance_factor(matrix):
    """Raíz simétrica para semidefinidas: conserva ceros por alias exactos.

    Rechaza negatividad material; recorta únicamente error de redondeo. No
    introduce un nugget que convertiría alias de una hoja en efectos distintos.
    """
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    if values.min(initial=0) < -1e-8:
        raise ValueError("Covarianza condicional no semidefinida positiva")
    return vectors * np.sqrt(np.maximum(values, 0))


def phylogenetic_design(covariance, training_tips, target_tips):
    """Factoriza hojas únicas y calcula predicción condicional conjunta.

    Si h_train = L z, entonces h_target = A z + B epsilon. Las filas repetidas
    de entrenamiento comparten exactamente el mismo efecto filogenético.
    """
    unique, inverse = np.unique(training_tips, return_inverse=True)
    chol = np.linalg.cholesky(covariance[np.ix_(unique, unique)])
    cross = covariance[np.ix_(target_tips, unique)]
    conditional_mean = solve_triangular(chol, cross.T, lower=True).T
    residual = (
        covariance[np.ix_(target_tips, target_tips)]
        - conditional_mean @ conditional_mean.T
    )
    return chol[inverse], conditional_mean, covariance_factor(residual), unique

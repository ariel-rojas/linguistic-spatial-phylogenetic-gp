"""Distribución estática de rendimiento al reservar cada familia por turno."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_distribution(metrics, folder):
    """Guarda histograma por familia y relación entre tamaño de prueba y exactitud.

    Incluye familias pequeñas: el segundo panel permite identificar resultados
    extremos basados en pocas lenguas. No supone independencia entre los ajustes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    accuracy = 100 * metrics.accuracy
    axes[0].hist(
        accuracy, bins=np.linspace(0, 100, 11), color="#2563eb", edgecolor="white"
    )
    axes[0].axvline(
        accuracy.mean(),
        color="#dc2626",
        linestyle="--",
        label=f"Media: {accuracy.mean():.1f}%",
    )
    axes[0].set(
        xlabel="Exactitud (%)",
        ylabel="Número de familias",
        xlim=(0, 100),
        title="Cada familia tiene el mismo peso",
    )
    axes[0].legend()
    axes[1].scatter(metrics.n_test, accuracy, color="#2563eb", alpha=0.65)
    axes[1].set(
        xscale="log",
        xlabel="Lenguas de prueba por familia (escala log)",
        ylabel="Exactitud (%)",
        ylim=(-3, 103),
        title="Tamaño de la familia y rendimiento",
    )
    fig.suptitle(
        f"Una familia fuera por turno · {len(metrics)} familias · {metrics.n_test.sum()} registros"
    )
    fig.savefig(folder / "distribution.png", dpi=160)
    plt.close(fig)

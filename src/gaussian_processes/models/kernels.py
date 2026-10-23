"""Composición aditiva de las estrategias espacial y filogenética."""

from .phylogenetic import BrownianPhylogeneticComponent
from .spatial import build_spatial_component


def build_kernel(config, covariance):
    """Suma componentes independientes usando la matriz filogenética preparada."""
    spatial = build_spatial_component(config)
    phylogenetic = BrownianPhylogeneticComponent(
        covariance, config["phylogenetic_rate"]
    )
    return spatial.build() + phylogenetic.build()

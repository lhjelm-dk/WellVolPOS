from .figures import (
    fig_a1_area_depth,
    fig_a2_outcome_tree,
    fig_a3_chance_decomposition,
    fig_a4_resource_vs_depth,
    fig_a5_exceedance,
    fig_a6_overlap,
    fig_b0_section,
    fig_b1_volume_split,
    fig_b2_chance_vs_regret,
    fig_b3_uncertainty_reduction,
)
from .theme import apply, colour, depth_axis, palette

__all__ = [
    "apply", "palette", "colour", "depth_axis",
    "fig_a1_area_depth", "fig_a2_outcome_tree", "fig_a3_chance_decomposition",
    "fig_a4_resource_vs_depth", "fig_a5_exceedance", "fig_a6_overlap",
    "fig_b0_section", "fig_b1_volume_split", "fig_b2_chance_vs_regret",
    "fig_b3_uncertainty_reduction",
]

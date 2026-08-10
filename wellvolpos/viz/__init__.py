"""Figures, in two backends driven from one theme.

``figures``     matplotlib -- the export path
``interactive`` plotly     -- what the app displays
``theme``       the palette, the panel height and the depth rule, shared by both

Import the plotly figures under their ``pfig_`` names so a call site always
says which backend it is using.
"""

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
    fig_b4_chance_waterfall,
    fig_b5_allocation_dumbbell,
    fig_b6_inverse,
)
from .interactive import (
    pfig_a1_area_depth,
    pfig_a2_outcome_tree,
    pfig_a3_chance_decomposition,
    pfig_a4_resource_vs_depth,
    pfig_a5_exceedance,
    pfig_a6_overlap,
    pfig_b0_section,
    pfig_b1_volume_split,
    pfig_b2_chance_vs_regret,
    pfig_b3_uncertainty_reduction,
    pfig_b4_chance_waterfall,
    pfig_b5_allocation_dumbbell,
    pfig_b6_inverse,
    pfig_map_view,
    row_zlim,
)
from .theme import (
    PANEL_HEIGHT,
    apply,
    apply_plotly,
    colour,
    depth_axis,
    depth_axis_plotly,
    palette,
)

__all__ = [
    "apply", "palette", "colour", "depth_axis",
    "apply_plotly", "depth_axis_plotly", "PANEL_HEIGHT", "row_zlim",
    # matplotlib / export
    "fig_a1_area_depth", "fig_a2_outcome_tree", "fig_a3_chance_decomposition",
    "fig_a4_resource_vs_depth", "fig_a5_exceedance", "fig_a6_overlap",
    "fig_b0_section", "fig_b1_volume_split", "fig_b2_chance_vs_regret",
    "fig_b3_uncertainty_reduction", "fig_b4_chance_waterfall", "fig_b5_allocation_dumbbell",
    "fig_b6_inverse",
    # plotly / interactive
    "pfig_a1_area_depth", "pfig_a2_outcome_tree", "pfig_a3_chance_decomposition",
    "pfig_a4_resource_vs_depth", "pfig_a5_exceedance", "pfig_a6_overlap",
    "pfig_b0_section", "pfig_b1_volume_split", "pfig_b2_chance_vs_regret",
    "pfig_b3_uncertainty_reduction", "pfig_b4_chance_waterfall", "pfig_b5_allocation_dumbbell",
    "pfig_b6_inverse", "pfig_map_view",
]

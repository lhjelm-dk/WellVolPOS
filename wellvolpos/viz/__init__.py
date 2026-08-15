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
    fig_b13_below_exit,
    fig_b2_chance_vs_regret,
    fig_b3_uncertainty_reduction,
    fig_b4_chance_waterfall,
    fig_b5_allocation_dumbbell,
    fig_b6_inverse,
    fig_b7_frontier,
    fig_b8_commercial_chance,
    fig_c1_section,
    fig_c2_exceedance,
    fig_c3_mefs_bars,
    fig_a8_contact_distribution,
    fig_a9_prospect_density,
    fig_b9_chance_weighted,
    fig_b11_pos_sensitivity,
    fig_b12_banded_percentiles,
)
from .interactive import (
    AREA_SCALES,
    pfig_a1_area_depth,
    pfig_a2_outcome_tree,
    pfig_a3_chance_decomposition,
    pfig_a4_resource_vs_depth,
    pfig_a5_exceedance,
    pfig_a6_overlap,
    pfig_b0_section,
    pfig_b1_volume_split,
    pfig_b13_below_exit,
    pfig_b2_chance_vs_regret,
    pfig_b3_uncertainty_reduction,
    pfig_b4_chance_waterfall,
    pfig_b5_allocation_dumbbell,
    pfig_b6_inverse,
    pfig_b7_frontier,
    pfig_b8_commercial_chance,
    CONCEPT_KEY,
    pfig_a8_contact_distribution,
    pfig_a9_prospect_density,
    pfig_b9_chance_weighted,
    pfig_b11_pos_sensitivity,
    pfig_b12_banded_percentiles,
    suggest_grid,
    pfig_c1_section,
    pfig_c2_exceedance,
    pfig_c3_mefs_bars,
    pfig_colour_key,
    pfig_map_view,
    row_zlim,
)
from .theme import (
    PANEL_HEIGHT,
    PROBABILITY_SCALES,
    TALL_PANEL_HEIGHT,
    VOLUME_SCALES,
    level_row,
    apply,
    apply_plotly,
    colour,
    depth_axis,
    depth_axis_plotly,
    palette,
)

__all__ = [
    "apply", "palette", "colour", "depth_axis",
    "apply_plotly", "depth_axis_plotly", "PANEL_HEIGHT", "level_row", "row_zlim", "AREA_SCALES",
    # matplotlib / export
    "fig_a1_area_depth", "fig_a2_outcome_tree", "fig_a3_chance_decomposition",
    "fig_a4_resource_vs_depth", "fig_a5_exceedance", "fig_a6_overlap",
    "fig_b0_section", "fig_b1_volume_split", "fig_b2_chance_vs_regret",
    "fig_b3_uncertainty_reduction", "fig_b4_chance_waterfall", "fig_b5_allocation_dumbbell",
    "fig_b6_inverse", "fig_b7_frontier", "fig_b8_commercial_chance", "fig_c1_section", "fig_c2_exceedance", "fig_c3_mefs_bars", "fig_a8_contact_distribution", "fig_a9_prospect_density", "fig_b9_chance_weighted", "fig_b11_pos_sensitivity",
    # plotly / interactive
    "pfig_a1_area_depth", "pfig_a2_outcome_tree", "pfig_a3_chance_decomposition",
    "pfig_a4_resource_vs_depth", "pfig_a5_exceedance", "pfig_a6_overlap",
    "pfig_b0_section", "pfig_b1_volume_split", "pfig_b2_chance_vs_regret",
    "pfig_b3_uncertainty_reduction", "pfig_b4_chance_waterfall", "pfig_b5_allocation_dumbbell",
    "pfig_b6_inverse", "pfig_b7_frontier", "pfig_b8_commercial_chance", "pfig_map_view", "pfig_c1_section", "pfig_c2_exceedance", "pfig_c3_mefs_bars", "pfig_a8_contact_distribution", "pfig_a9_prospect_density", "pfig_b9_chance_weighted", "pfig_b11_pos_sensitivity", "suggest_grid", "pfig_colour_key", "CONCEPT_KEY",
]

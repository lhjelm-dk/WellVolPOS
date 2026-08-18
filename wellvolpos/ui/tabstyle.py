"""A light fill per tab, so the six are told apart at a glance.

Lars asked for this on 2026-08-12. The problem it solves is real: six tabs of
plain text differ only by a label, and the label is the first thing that goes when
someone is scrolling a long tab and loses their place.

**The colours are navigation, not meaning.** They are deliberately *not* drawn from
``theme.LIGHT``: every colour in that palette is a volume concept, and reusing one
here would say a tab was "about" that volume. These are muted seaborn-ish pastels
chosen only to be distinct from each other and pale enough that black label text
stays legible on them -- which is why they carry no docstring claim about what any
tab contains.

Injected as CSS because Streamlit exposes no per-tab styling. It therefore depends on
Streamlit's DOM, so it is isolated in its own module: if an upgrade changes the
markup the tabs lose their tint and nothing else breaks.

The selectors target **ARIA roles** -- ``[role="tablist"] [role="tab"]`` -- not
baseweb's ``data-baseweb`` attributes. The first attempt used the latter and styled
nothing at all, because this Streamlit renders tabs as ``div[role=tab]`` with no
baseweb attribute. Roles are part of the accessibility contract and change far less
often than internal component names.
"""

from __future__ import annotations

import streamlit as st

#: One pale fill per tab, in tab order. Seaborn's "pastel" family, lightened --
#: distinct in hue *and* close in lightness, so no tab shouts louder than another.
TAB_FILLS = (
    "#dbe7f3",   # ① data, QC and risk      -- pale blue
    "#e2eddc",   # ② prospect               -- pale green
    "#fdeacd",   # ③ where to drill         -- pale amber
    "#f7dcdc",   # ④ at this well           -- pale rose
    "#e8e0f0",   # ⑤ risk & report          -- pale lilac
    "#e9e6df",   # ⑥ theory & guide         -- warm grey
)

#: Tab label colours. Grey, both of them -- see the note in :func:`inject` for why
#: the accent red is not used for the selected tab.
ACTIVE_TEXT = "#2b2a28"
INACTIVE_TEXT = "#52514e"

__all__ = ["ACTIVE_TEXT", "INACTIVE_TEXT", "TAB_FILLS", "inject"]


def inject() -> None:
    """Write the per-tab CSS. Call once, after ``st.tabs``."""
    rules = [
        '[role="tablist"] { gap: 4px !important; }',
        '[role="tablist"] [role="tab"] {'
        ' border-radius: 8px 8px 0 0 !important;'
        ' padding: 8px 18px !important;'
        ' border-bottom: none !important; }',
        # **Bold grey, never the accent red** (Lars, 2026-08-12). Streamlit paints the
        # selected tab in its primary colour -- rgb(255,75,75) -- on the tab div, on
        # the inner <p>, and on the bottom border. All three are overridden: red is
        # the only colour in this app that means something (a threshold volume), and
        # spending it on "you are here" would make the tab strip argue with every
        # MEFS line on the page.
        #
        # Which tab is *active* is therefore shown by weight and full saturation of
        # its own fill; the hue only says which tab it is. Two questions, two cues.
        f'[role="tablist"] [role="tab"][aria-selected="true"],'
        f' [role="tablist"] [role="tab"][aria-selected="true"] p {{'
        f' color: {ACTIVE_TEXT} !important; font-weight: 700 !important;'
        f' border-bottom-color: transparent !important; }}',
        f'[role="tablist"] [role="tab"][aria-selected="false"],'
        f' [role="tablist"] [role="tab"][aria-selected="false"] p {{'
        f' color: {INACTIVE_TEXT} !important; font-weight: 500 !important; }}',
        '[role="tablist"] [role="tab"][aria-selected="false"] { opacity: 0.62 !important; }',
        # The sliding underline is a separate react-aria element, not the tab's own
        # border -- which is why overriding border-bottom-color left a red bar
        # behind. Recoloured rather than hidden: it is a useful third cue for where
        # you are, it just must not be the accent red.
        f'.react-aria-SelectionIndicator {{ background-color: {ACTIVE_TEXT} !important; }}',
    ]
    for i, fill in enumerate(TAB_FILLS, start=1):
        rules.append(
            f'[role="tablist"] [role="tab"]:nth-child({i})'
            f' {{ background-color: {fill} !important; }}'
        )
    st.markdown("<style>" + "\n".join(rules) + "</style>", unsafe_allow_html=True)


#: Metric values, one size down. The default is a headline size, which is right for one
#: number and wrong for a strip of eight -- and every strip in this app is a strip.
METRIC_CSS = """
<style>
  [data-testid="stMetricValue"] { font-size: 1.35rem; line-height: 1.25; }
  [data-testid="stMetricLabel"] p { font-size: 0.80rem; }
</style>
"""


def apply_metric_size() -> None:
    """Shrink metric values. Call once, beside the tab tint."""
    st.markdown(METRIC_CSS, unsafe_allow_html=True)

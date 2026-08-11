"""The area-depth curve A(z), recovered from the trials themselves.

In an area x gross-pay volumetric model the productive area is the map-view area
enclosed by the contact, so the (contact, area) pairs across the trial set trace
out the closure's area-depth curve directly. On the reference dataset the fit is
essentially exact -- isotonic R2 = 0.9999999987, residual SD 0.000 km2 -- because
GeoX evaluates area as a deterministic function of contact depth.

That matters because A(z) is what converts a well's *depth* into a well's
*position on the structure*, and therefore what lets a trial be split at the
well. Where the fit is poor the split is not trustworthy, so the fit quality is
reported rather than assumed: see :attr:`AreaDepth.r2`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class Contours:
    """One set of map contours: where they are, how big, and how trustworthy.

    ``extrapolated`` marks contours shallower than the shallowest *sampled*
    contact, whose area comes from the taper in
    :meth:`AreaDepth.area_at_tapered` rather than from any trial.
    ``at_data_limit`` marks the deepest ring, which is the base of the sampled
    data rather than a round contour -- carried as a flag so a figure can label
    it as such instead of inferring it from position.
    """

    depths: np.ndarray
    radii: np.ndarray
    extrapolated: np.ndarray
    at_data_limit: np.ndarray

    def __iter__(self):
        """Unpack as ``depths, radii, extrapolated`` for the common case."""
        return iter((self.depths, self.radii, self.extrapolated))


@dataclass
class AreaDepth:
    """Monotone area-depth curve with inverse lookup.

    Deeper contact -> larger enclosed area, so the fit is constrained to be
    non-decreasing. That constraint is geological, not cosmetic: an unconstrained
    smoother can produce a locally decreasing curve that would make a trial's
    up-dip fraction exceed 1.
    """

    z: np.ndarray            # depth grid, increasing (m TVDSS)
    a: np.ndarray            # area on that grid (km2), non-decreasing
    r2: float
    resid_sd: float
    n_points: int

    # -------------------------------------------------------------- builders
    @classmethod
    def from_trials(cls, contact, area, *, grid: int = 2000) -> "AreaDepth":
        c = np.asarray(contact, dtype=float)
        ar = np.asarray(area, dtype=float)
        ok = np.isfinite(c) & np.isfinite(ar) & (ar > 0)
        if ok.sum() < 10:
            raise ValueError(
                f"need at least 10 trials with positive area to fit A(z); got {int(ok.sum())}"
            )
        c, ar = c[ok], ar[ok]
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(c, ar)
        pred = iso.predict(c)
        resid = ar - pred
        var = float(np.var(ar))
        r2 = 1.0 - float(np.var(resid)) / var if var > 0 else float("nan")
        zg = np.linspace(float(c.min()), float(c.max()), grid)
        ag = np.maximum.accumulate(iso.predict(zg))
        return cls(z=zg, a=ag, r2=r2, resid_sd=float(np.std(resid)), n_points=int(ok.sum()))

    # -------------------------------------------------------------- lookups
    def area_at(self, depth):
        """Enclosed area (km2) at a structural level. Clipped outside the data."""
        return np.interp(depth, self.z, self.a)

    def depth_at(self, area):
        """Inverse: structural level (m TVDSS) enclosing a given area."""
        return np.interp(area, self.a, self.z)

    @property
    def shallowest(self) -> float:
        return float(self.z[0])

    @property
    def deepest(self) -> float:
        return float(self.z[-1])

    def apex_estimate(self, *, window_m: float = 60.0) -> float:
        """Linear extrapolation of the shallow tail to A = 0.

        This is a *convenience only*. The apex is a mapped quantity the user
        knows; extrapolating it from the shallow tail of a fitted curve is an
        estimate whose error is unbounded when the trial set does not reach far
        enough up the structure. Always prefer a user-supplied apex, and treat
        this as a starting value in the UI, not a result.
        """
        m = self.z < self.z[0] + window_m
        if m.sum() < 3:
            return float(self.z[0])
        slope, intercept = np.polyfit(self.z[m], self.a[m], 1)
        if slope <= 0:
            return float(self.z[0])
        return float(-intercept / slope)

    def area_at_tapered(self, depth, apex: float) -> np.ndarray:
        """A(z), but tapering linearly to zero at ``apex`` above the sampled range.

        :meth:`area_at` is ``np.interp``, which *clips*: asked for a depth above
        the shallowest sampled contact it keeps returning that contact's area.
        For a map view that is plainly wrong -- every contour above the
        shallowest sampled contact came out the same size, so the closure
        appeared to have a wide flat top where in fact the trials simply never
        reached the crest. Between the apex and the shallowest sampled contact
        the area is unknown, and a straight taper to zero at the apex is the
        least-committal thing to draw; callers are expected to mark that stretch
        as extrapolated.
        """
        z = np.asarray(depth, dtype=float)
        a = np.asarray(self.area_at(z), dtype=float)
        top = self.shallowest
        if top > float(apex):
            frac = np.clip((z - float(apex)) / (top - float(apex)), 0.0, 1.0)
            a = np.where(z < top, float(self.a[0]) * frac, a)
        a = np.where(z <= float(apex), 0.0, a)
        return a

    def contour_radii(
        self, apex: float, *, interval: float = 50.0, z_max: float | None = None
    ) -> "Contours":
        """Contours for a map view, on round absolute depths.

        The depths are the **multiples of** ``interval`` that fall inside the
        closure -- 3250, 3300, 3350 ... for a 50 m interval; 3300, 3400, 3500 for
        100 m -- from the first multiple below ``apex`` down to the last one above
        ``z_max`` (default: the deepest depth A(z) covers). The deepest depth
        itself is then appended as one further ring, because it is the base of the
        sampled data and worth seeing even though it will not be a round number.

        Contours are *not* stepped off the apex, and that is the point. The apex
        is an estimate -- extrapolated from A(z)'s shallow tail unless the user
        supplies a mapped value -- so apex-relative contours move every time it is
        nudged, and no two runs are comparable. Round absolute depths do not move:
        changing the apex changes only *which* contours fall inside the closure.
        It is also how a depth map is read, against seismic and a prognosis on the
        same datum; 3268.3 m is not a contour anyone would draw.

        A consequence, and the right one: the innermost gap is usually a partial
        interval, since the shallowest round contour sits a little below the apex
        rather than exactly one interval down. A contour map at a crest does the
        same.

        The radius is ``sqrt(A(z) / pi)`` -- the radius a circle of that enclosed
        area would have. **This is a cartoon, not a map.** A(z) records how much
        area each contact encloses and says nothing whatever about the shape of
        the closure or where the thickest part is, so a real prospect is never
        this set of nested circles. What the picture does carry faithfully is the
        *area* at each depth, and therefore the spacing between contours -- which
        is what makes it useful for seeing how much of the closure sits above a
        proposed well.
        """
        hi = self.deepest if z_max is None else float(z_max)
        step = float(interval)
        if step <= 0:
            raise ValueError("contour interval must be positive")

        # First multiple of `step` strictly deeper than the apex. floor()+1 rather
        # than ceil(), so an apex that lands exactly on a multiple still gets its
        # first contour one interval down -- a contour at the apex encloses no
        # area and would draw as a point.
        first_k = int(np.floor(float(apex) / step)) + 1
        last_k = int(np.floor(hi / step))
        rounds = np.arange(first_k, last_k + 1, dtype=float) * step
        rounds = rounds[(rounds > float(apex)) & (rounds <= hi)]

        # The base of the data, kept as its own ring unless a round contour
        # already lands on it.
        if rounds.size == 0 or abs(rounds[-1] - hi) > 1e-9:
            depths = np.append(rounds, hi)
            at_limit = np.zeros(depths.size, dtype=bool)
            at_limit[-1] = True
        else:
            depths = rounds
            at_limit = np.zeros(depths.size, dtype=bool)
            at_limit[-1] = True

        radii = np.sqrt(np.maximum(self.area_at_tapered(depths, apex), 0.0) / np.pi)
        return Contours(
            depths=depths, radii=radii,
            extrapolated=depths < self.shallowest, at_data_limit=at_limit,
        )

    def radius_at(self, depth: float, apex: float) -> float:
        """Equivalent-circle radius (km) at one structural level."""
        return float(np.sqrt(max(float(self.area_at_tapered(depth, apex)), 0.0) / np.pi))

    # ------------------------------------------------- closure bulk volume
    #: How the closure volume is integrated between two contours.
    #:
    #: ``"trapezoid"`` -- ``(A1 + A2)/2 x h``. The default, and validated: it is what
    #: makes ``core.reservoir`` recover GeoX's own reservoir-thickness column to a mean
    #: difference of 0.01 m on prospect A and 0.01 m on prospect B, two independent
    #: files. Whatever GeoX does internally, this reproduces it.
    #:
    #: ``"frustum"`` -- ``h/3 x (A1 + A2 + sqrt(A1 x A2))``, the pyramidal rule, which
    #: is the one Lars's 2018 workbook uses in column ``BK``. It is *exact* for a cone
    #: or pyramid where the trapezoid rule over-estimates, so it is the better rule in
    #: principle. Offered rather than imposed: switching the default would move the
    #: validated thickness inversion, and on these closures the difference is far
    #: smaller than the apex extrapolation error it would sit inside. See
    #: ``tests/test_structure.py`` for the measured gap.
    VOLUME_RULES = ("trapezoid", "frustum")

    def _volume_grid(self, apex: float, n: int = 4000, rule: str = "trapezoid"
                     ) -> tuple[np.ndarray, np.ndarray]:
        """(depths, cumulative volume above each depth) from the apex down.

        ``km2 x m``, which is ``1e6 m3`` -- the same unit GeoX writes HC-bearing
        gross rock volume in, so the two are directly comparable with no factor.

        See :data:`VOLUME_RULES` for the two integration rules and why the
        trapezoid one is the default despite the frustum rule being exact on a cone.
        """
        if rule not in self.VOLUME_RULES:
            raise ValueError(f"unknown volume rule {rule!r}; expected one of {self.VOLUME_RULES}")
        z = np.linspace(float(apex), self.deepest, int(n))
        a = self.area_at_tapered(z, apex)
        h = np.diff(z)
        if rule == "frustum":
            a1, a2 = a[:-1], a[1:]
            dv = h / 3.0 * (a1 + a2 + np.sqrt(np.maximum(a1 * a2, 0.0)))
        else:
            dv = 0.5 * (a[1:] + a[:-1]) * h
        return z, np.concatenate([[0.0], np.cumsum(dv)])

    def volume_above(self, depth, apex: float, rule: str = "trapezoid") -> np.ndarray:
        """Bulk closure volume above ``depth``: the integral of A(z) from the apex.

        This is the volume a reservoir of unlimited thickness would enclose above
        that level -- the ceiling on any hydrocarbon-bearing GRV with the contact
        there.
        """
        z, v = self._volume_grid(apex, rule=rule)
        return np.interp(depth, z, v)

    def depth_for_volume(self, volume, apex: float, rule: str = "trapezoid") -> np.ndarray:
        """Inverse of :meth:`volume_above`: the depth enclosing a given volume."""
        z, v = self._volume_grid(apex, rule=rule)
        return np.interp(volume, v, z)

    def quality(self) -> tuple[str, str]:
        """(level, message) for the QC report."""
        if not np.isfinite(self.r2):
            return "fail", "A(z) could not be fitted."
        if self.r2 >= 0.99:
            return "pass", f"A(z) fit R2 = {self.r2:.6f}, residual SD {self.resid_sd:.4f} km2."
        if self.r2 >= 0.90:
            return (
                "warn",
                f"A(z) fit R2 = {self.r2:.4f}, residual SD {self.resid_sd:.4f} km2. Area is only "
                f"loosely determined by contact depth, so the per-trial split is approximate.",
            )
        return (
            "fail",
            f"A(z) fit R2 = {self.r2:.4f}. Area is not a function of contact depth in this model; "
            f"the proven/possible split cannot be trusted. Use the reference grouping engine.",
        )

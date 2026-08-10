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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Depths, equivalent-circle radii and an extrapolated flag, for a map view.

        Returns ``(depths, radii_km, extrapolated)`` from the first contour below
        ``apex`` down to ``z_max`` (default: the deepest depth A(z) covers),
        spaced by ``interval``. ``extrapolated`` is True for contours shallower
        than the shallowest *sampled* contact, where the area comes from the
        taper in :meth:`area_at_tapered` rather than from any trial.

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
        first = float(apex) + step
        if first > hi:
            depths = np.array([hi], dtype=float)
        else:
            depths = np.arange(first, hi + 0.5 * step, step, dtype=float)
            depths = depths[depths <= hi]
            if depths.size == 0 or depths[-1] < hi - 1e-9:
                depths = np.append(depths, hi)
        radii = np.sqrt(np.maximum(self.area_at_tapered(depths, apex), 0.0) / np.pi)
        return depths, radii, depths < self.shallowest

    def radius_at(self, depth: float, apex: float) -> float:
        """Equivalent-circle radius (km) at one structural level."""
        return float(np.sqrt(max(float(self.area_at_tapered(depth, apex)), 0.0) / np.pi))

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

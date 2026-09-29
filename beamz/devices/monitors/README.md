A Monitor is a functional mapping

𝑀
:
(
𝐸
,
𝐻
,
𝑡
)
→
observable
(
𝑡
)
M:(E,H,t)→observable(t)

Implementation:

class Monitor:
    def __init__(self, region, op, temporal_reduction=None, spatial_reduction=None):
        self.region = region          # bounding box or surface
        self.op = op                  # function on E,H per cell
        self.temporal_reduction = temporal_reduction  # "instant", "DFT", "accumulate"
        self.spatial_reduction = spatial_reduction    # "none", "integrate", "average"


op could be a lambda or GPU shader snippet like dot(E, H) or abs(E)**2.

temporal_reduction defines whether to store time-series, accumulate FFTs, or just instantaneous values.

spatial_reduction defines whether to integrate or store field maps.

Then:

“flux monitor” = op = cross(E,H).n, temporal=DFT, spatial=integrate

“field recorder” = selected field components, temporal=interval, spatial=none

“mode monitor” = op = dot(E, E_mode*), temporal=DFT, spatial=integrate

“energy monitor” = op = |E|^2 + |H|^2, temporal=accumulate, spatial=integrate

All the same code path, just different parameters.

For a 3D flux plane on a planar material interface, `FluxMonitor` accepts
`colocation_side="plus"` or `"minus"`. Tangential H is reconstructed at the
unchanged plane with a quadratic polynomial through three Yee samples from the
chosen normal side; E retains linear interpolation. Select a side that remains
in one smooth medium across those samples, preferably the better resolved medium.
This avoids crossing a derivative discontinuity in H, but does not remove spatial
field-update error. The default `None` retains symmetric linear interpolation.
The option is included in immutable monitor identity and is rejected for 2D
simulation grids. No new dynamical state or constitutive approximation is added.

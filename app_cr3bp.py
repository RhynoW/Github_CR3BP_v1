#!/usr/bin/env python3
"""
app_cr3bp.py — Streamlit GUI for the CR3BP / Cislunar / Interplanetary simulator.
Supports Expert Mode (full controls) and Teaching Mode (3-step wizard).

Launch with:
    streamlit run app_cr3bp.py
"""
from __future__ import annotations

import datetime
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Load simulation engine ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import gmat_intercept_sim_v2_CR3BP as sim

MU_STAR = sim.MU_STAR
D_EM    = sim.D_EM
MU_MARS = sim.MU_MARS
R_MARS  = sim.R_MARS

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Open Mission Ops Lab",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricDelta"] { font-size: 0.8rem; }
.edu-banner {
    background: linear-gradient(135deg, #effd5f, #e47200);
    border-left: 4px solid #4a9eff;
    padding: 12px 18px; border-radius: 6px; margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

if "mission_log" not in st.session_state:
    st.session_state["mission_log"] = []
if "edu_step" not in st.session_state:
    st.session_state["edu_step"] = 0
if "fr_edu_computed" not in st.session_state:
    st.session_state["fr_edu_computed"] = False
if "fr_exp_computed" not in st.session_state:
    st.session_state["fr_exp_computed"] = False


# ─────────────────────────────────────────────────────────────────────────────
# Scene / colour constants
# ─────────────────────────────────────────────────────────────────────────────

_DARK = "#0e1117"
_GRID = "#1f2535"

_ENG_BG   = "#070d1a"
_ENG_GRID = "#1a3050"
_ENG_SCENE = dict(
    bgcolor=_ENG_BG,
    xaxis=dict(gridcolor=_ENG_GRID, backgroundcolor=_ENG_BG, showbackground=True),
    yaxis=dict(gridcolor=_ENG_GRID, backgroundcolor=_ENG_BG, showbackground=True),
    zaxis=dict(gridcolor=_ENG_GRID, backgroundcolor=_ENG_BG, showbackground=True),
)

_STAR_BG   = "#000008"
_STAR_GRID = "#0d0d28"
_STAR_SCENE = dict(
    bgcolor=_STAR_BG,
    xaxis=dict(gridcolor=_STAR_GRID, backgroundcolor=_STAR_BG, showbackground=True),
    yaxis=dict(gridcolor=_STAR_GRID, backgroundcolor=_STAR_BG, showbackground=True),
    zaxis=dict(gridcolor=_STAR_GRID, backgroundcolor=_STAR_BG, showbackground=True),
)

_LAYOUT_BASE = dict(
    paper_bgcolor=_DARK, font_color="white",
    margin=dict(l=0, r=60, b=0, t=36),
    legend=dict(bgcolor="rgba(20,24,40,0.85)", font_color="white"),
)
_CAMERA = dict(eye=dict(x=1.5, y=1.5, z=0.8))


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sphere(cx: float, cy: float, cz: float, r: float, n: int = 20):
    u = np.linspace(0, 2 * math.pi, n)
    v = np.linspace(0, math.pi, n)
    x = cx + r * np.outer(np.cos(u), np.sin(v))
    y = cy + r * np.outer(np.sin(u), np.sin(v))
    z = cz + r * np.outer(np.ones(n), np.cos(v))
    return x, y, z


def _thin(lst: list, max_pts: int = 3000) -> list:
    n = len(lst)
    if n <= max_pts:
        return lst
    idx = np.round(np.linspace(0, n - 1, max_pts)).astype(int)
    return [lst[i] for i in idx]


def _hohmann_v(alt_leo_km: float) -> float:
    """TLI speed for a standard Hohmann transfer from LEO to the Moon's orbit."""
    r = sim.R_EARTH + alt_leo_km
    return float(math.sqrt(sim.MU_EARTH * (2.0 / r - 1.0 / (0.5 * (r + D_EM)))))


def _nav_buttons(back_label: Optional[str] = None, back_step: int = 0,
                  fwd_label:  Optional[str] = None, fwd_step:  int = 1) -> None:
    """Render wizard back / forward navigation buttons."""
    pairs = [(lbl, step) for lbl, step in
             [(back_label, back_step), (fwd_label, fwd_step)] if lbl is not None]
    if not pairs:
        return
    cols = st.columns(len(pairs))
    for col, (lbl, step) in zip(cols, pairs):
        if col.button(lbl, use_container_width=True):
            st.session_state["edu_step"] = step
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Cached computation
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_lagrange_jacobi() -> Dict[str, float]:
    lp = sim.CR3BPPropagator.lagrange_points()
    out: Dict[str, float] = {}
    for name, pos in lp.items():
        x, y = float(pos[0]), float(pos[1])
        r1 = math.sqrt((x + MU_STAR) ** 2 + y ** 2 + 1e-30)
        r2 = math.sqrt((x - 1 + MU_STAR) ** 2 + y ** 2 + 1e-30)
        U  = 0.5 * (x ** 2 + y ** 2) + (1 - MU_STAR) / r1 + MU_STAR / r2
        out[name] = 2.0 * U
    return out


@st.cache_data(show_spinner=False)
def run_cr3bp(alt_leo_km: float, v_boost_kms: float,
              sim_days: float, n_steps: int):
    prop  = sim.CR3BPPropagator()
    r_geo = np.array([sim.R_EARTH + alt_leo_km, 0.0, 0.0])
    v_geo = np.array([0.0, v_boost_kms, 0.0])
    s0    = prop.eci_to_cr3bp(r_geo, v_geo, t_s=0.0)
    hist  = prop.propagate(s0, t_final_days=sim_days, n_steps=n_steps)
    lp    = prop.lagrange_points()
    return hist, lp


@st.cache_data(show_spinner=False)
def run_plan(alt_leo_km: float, alt_lmo_km: float):
    return sim.CislunarPlanner().plan(alt_leo_km=alt_leo_km, alt_lmo_km=alt_lmo_km)


@st.cache_data(show_spinner=False)
def run_free_return(alt_leo_km: float, periselene_km: float = 6500) -> tuple:
    """
    Bisect for TLI speed giving lunar periselene ≈ periselene_km, propagate 20 days.
    Returns (history, v_tli_kms, actual_periselene_km).

    Launch epoch: t_s = -10 days from J2000, when Moon is at ~86° ECI.
    At this epoch, TLI velocity [0, v, 0] maps to CR3BP vx_rot ≈ v*sin(86°)/V* ≈ v/V*,
    pointing nearly straight at the Moon.  At t_s=0 (Moon at 218°) vx_rot is negative
    (away from Moon); at t_s=+11d (Moon at 3°) vx_rot is only ~0.06*v/V* (mostly y).
    Neither of those converges.

    The min-Moon-distance function is V-shaped in v_TLI with nadir ≈ 5 000 km near
    v_hoh+0.006.  Bracket [v_hoh+0.003, v_hoh+0.0066] straddles the solution reliably.

    20-day window shows the complete figure-8: outbound leg (upper arc → Moon flyby)
    and return leg (lower arc through L5 region → Earth perigee at t≈+6.5d).
    The periselene is near-side of Moon (x slightly below x_moon) — the spacecraft
    approaches from the far side (x > x_moon), swings through periselene, and returns.
    """
    _T_LAUNCH_S = -10.0 * 86400.0   # Moon ≈ 86° ECI — TLI velocity nearly toward Moon

    prop = sim.CR3BPPropagator()

    def _eval(v_tli: float):
        r0 = np.array([sim.R_EARTH + alt_leo_km, 0.0, 0.0])
        v0 = np.array([0.0, v_tli, 0.0])
        s0 = prop.eci_to_cr3bp(r0, v0, t_s=_T_LAUNCH_S)
        hist = prop.propagate(s0, t_final_days=20.0, n_steps=12000)
        return min(s.dist_to_moon_km() for s in hist), hist

    v_hoh = _hohmann_v(alt_leo_km)
    # v_lo → min_dist ≈ 35 000 km (> target); v_hi → min_dist ≈ 4 000 km (< target).
    # Both straddle the V-minimum; bisection converges in ~10 iterations.
    v_lo, v_hi = v_hoh + 0.003, v_hoh + 0.0066
    best_hist: list = []
    best_v = 0.5 * (v_lo + v_hi)

    for _ in range(30):
        v_mid = 0.5 * (v_lo + v_hi)
        md, hist = _eval(v_mid)
        if md > periselene_km:
            v_lo = v_mid
        else:
            v_hi = v_mid
            best_hist = hist
            best_v = v_mid
        if (v_hi - v_lo) < 5.0e-5:
            break

    if not best_hist:
        _, best_hist = _eval(0.5 * (v_lo + v_hi))
        best_v = 0.5 * (v_lo + v_hi)

    actual_peri = min(s.dist_to_moon_km() for s in best_hist)
    return best_hist, round(best_v, 5), round(actual_peri, 1)


@st.cache_data(show_spinner=False)
def run_hohmann_cr3bp(alt_leo_km: float) -> list:
    """
    Hohmann TLI trajectory using the same t_s=-10d epoch as run_free_return,
    so both trajectories start from the same rotating-frame position for comparison.
    Extended to 20 days (same as free-return) to show that Hohmann drifts away
    with no return — contrasting with the free-return figure-8 shape.
    """
    prop = sim.CR3BPPropagator()
    r0 = np.array([sim.R_EARTH + alt_leo_km, 0.0, 0.0])
    v0 = np.array([0.0, _hohmann_v(alt_leo_km), 0.0])
    s0 = prop.eci_to_cr3bp(r0, v0, t_s=-10.0 * 86400.0)
    return prop.propagate(s0, t_final_days=20.0, n_steps=12000)


@st.cache_data(show_spinner=False)
def run_porkchop(dep_start: float, dep_end: float,
                 tof_min: float, tof_max: float,
                 dep_n: int, tof_n: int) -> sim.PorkchopGrid:
    return sim.porkchop(
        dep_j2000_days_start=dep_start, dep_j2000_days_end=dep_end,
        dep_n=dep_n, tof_min_days=tof_min, tof_max_days=tof_max, tof_n=tof_n,
        alt_park_km=200.0, alt_arr_km=400.0, verbose=False,
    )


@st.cache_data(show_spinner=False)
def run_single_leg(dep_day: float, tof_day: float) -> Optional[sim.InterplanetaryLeg]:
    return sim.InterplanetaryPlanner(200.0, 400.0).compute_leg(
        dep_day * 86400.0, tof_day * 86400.0
    )


# ─────────────────────────────────────────────────────────────────────────────
# CZML export
# ─────────────────────────────────────────────────────────────────────────────

def _j2000_to_iso(t_s: float) -> str:
    j2000 = datetime.datetime(2000, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    return (j2000 + datetime.timedelta(seconds=t_s)).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_czml_cislunar(history: List[sim.CR3BPState]) -> bytes:
    prop  = sim.CR3BPPropagator()
    thin  = _thin(history, max_pts=600)
    epoch = "2000-01-01T12:00:00Z"
    carts: List[float] = []
    for s in thin:
        r_km, _ = prop.cr3bp_to_eci(s)
        carts.extend([float(s.t_s),
                       float(r_km[0]) * 1e3,
                       float(r_km[1]) * 1e3,
                       float(r_km[2]) * 1e3])
    doc = [
        {"id": "document", "name": "CR3BP Cislunar Trajectory", "version": "1.0"},
        {
            "id": "spacecraft", "name": "Spacecraft",
            "availability": f"{_j2000_to_iso(thin[0].t_s)}/{_j2000_to_iso(thin[-1].t_s)}",
            "position": {
                "epoch": epoch, "referenceFrame": "INERTIAL",
                "interpolationAlgorithm": "LAGRANGE", "interpolationDegree": 5,
                "cartesian": carts,
            },
            "path": {
                "material": {"solidColor": {"color": {"rgba": [0, 200, 255, 200]}}},
                "width": 2.0, "leadTime": 0, "trailTime": 86400.0 * 30,
            },
        },
    ]
    return json.dumps(doc).encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Plot — 3D rotating frame (engineering grid)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rotating(history: List[sim.CR3BPState],
                  lp_dict: Dict[str, np.ndarray],
                  scrub_idx: Optional[int] = None,
                  show_soi: bool = True) -> go.Figure:
    hist = _thin(history)
    xs = [s.pos[0] for s in hist]
    ys = [s.pos[1] for s in hist]
    zs = [s.pos[2] for s in hist]
    ts = [s.t_days  for s in hist]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=ts, colorscale="Viridis", width=4,
                  colorbar=dict(title="天", thickness=10, len=0.55, x=1.04)),
        name="軌跡",
    ))

    if scrub_idx is not None:
        s = history[scrub_idx]
        fig.add_trace(go.Scatter3d(
            x=[s.pos[0]], y=[s.pos[1]], z=[s.pos[2]], mode="markers",
            marker=dict(size=10, color="#ff4b4b", symbol="diamond"),
            name=f"SC  t={s.t_days:.2f}d",
        ))

    xe, ye, ze = _sphere(-MU_STAR, 0, 0, sim.R_EARTH / D_EM)
    fig.add_trace(go.Surface(x=xe, y=ye, z=ze, colorscale="Blues",
                              showscale=False, name="Earth", opacity=0.85,
                              hoverinfo="skip"))

    xm, ym, zm = _sphere(1 - MU_STAR, 0, 0, sim.R_MOON / D_EM)
    fig.add_trace(go.Surface(x=xm, y=ym, z=zm,
                              colorscale=[[0, "#777"], [1, "#ddd"]],
                              showscale=False, name="Moon", opacity=0.85,
                              hoverinfo="skip"))

    if show_soi:
        r_soi_norm = sim.BODIES["Moon"]["soi_km"] / D_EM
        xs_s, ys_s, zs_s = _sphere(1 - MU_STAR, 0, 0, r_soi_norm, n=28)
        fig.add_trace(go.Surface(
            x=xs_s, y=ys_s, z=zs_s,
            colorscale=[[0, "rgba(140,170,255,0.06)"], [1, "rgba(140,170,255,0.06)"]],
            showscale=False, name=f"月球 SOI ({sim.BODIES['Moon']['soi_km']:.0f} km)",
            opacity=0.10, hoverinfo="skip",
        ))

    for name, pos in lp_dict.items():
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode="markers+text", text=[name], textposition="top center",
            marker=dict(size=5, color="gold", symbol="diamond"),
            name=name, showlegend=(name in ("L1", "L2", "L4", "L5")),
        ))

    z_max = max(0.05, max(abs(z) for z in zs)) * 1.6
    sc = {
        "xaxis": dict(**_ENG_SCENE["xaxis"], title="x (norm.)", range=[-1.5, 1.5]),
        "yaxis": dict(**_ENG_SCENE["yaxis"], title="y (norm.)", range=[-1.2, 1.2]),
        "zaxis": dict(**_ENG_SCENE["zaxis"], title="z (norm.)", range=[-z_max, z_max]),
        "bgcolor": _ENG_BG, "aspectmode": "data",
    }
    fig.update_layout(
        **_LAYOUT_BASE, scene=sc, height=520,
        scene_camera=_CAMERA,
        title=dict(text="CR3BP — 旋轉座標系 (工程格)", font_color="white"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot — 3D inertial frame (starfield)
# ─────────────────────────────────────────────────────────────────────────────

def plot_inertial(history: List[sim.CR3BPState],
                  scrub_idx: Optional[int] = None) -> go.Figure:
    prop = sim.CR3BPPropagator()
    hist = _thin(history)
    eci  = [prop.cr3bp_to_eci(s) for s in hist]
    xs = [p[0][0] / D_EM for p in eci]
    ys = [p[0][1] / D_EM for p in eci]
    zs = [p[0][2] / D_EM for p in eci]
    ts = [s.t_days        for s in hist]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=ts, colorscale="Plasma", width=4,
                  colorbar=dict(title="天", thickness=10, len=0.55, x=1.04)),
        name="軌跡 (ECI)",
    ))

    if scrub_idx is not None:
        r_e, _ = prop.cr3bp_to_eci(history[scrub_idx])
        fig.add_trace(go.Scatter3d(
            x=[r_e[0] / D_EM], y=[r_e[1] / D_EM], z=[r_e[2] / D_EM],
            mode="markers", marker=dict(size=10, color="#ff4b4b"),
            name=f"SC  t={history[scrub_idx].t_days:.2f}d",
        ))

    xe, ye, ze = _sphere(0, 0, 0, sim.R_EARTH / D_EM)
    fig.add_trace(go.Surface(x=xe, y=ye, z=ze, colorscale="Blues",
                              showscale=False, name="Earth", opacity=0.85,
                              hoverinfo="skip"))

    step = max(1, len(hist) // 80)
    moon_eci = [sim.SimpleEphem.moon_eci(s.t_s) for s in hist[::step]]
    fig.add_trace(go.Scatter3d(
        x=[p[0][0] / D_EM for p in moon_eci],
        y=[p[0][1] / D_EM for p in moon_eci],
        z=[p[0][2] / D_EM for p in moon_eci],
        mode="lines", line=dict(color="silver", width=1, dash="dot"),
        name="月球軌道弧",
    ))

    sc = {
        "xaxis": dict(**_STAR_SCENE["xaxis"], title="x/D_EM (ECI)"),
        "yaxis": dict(**_STAR_SCENE["yaxis"], title="y/D_EM (ECI)"),
        "zaxis": dict(**_STAR_SCENE["zaxis"], title="z/D_EM (ECI)"),
        "bgcolor": _STAR_BG, "aspectmode": "data",
    }
    fig.update_layout(
        **_LAYOUT_BASE, scene=sc, height=520,
        scene_camera=_CAMERA,
        title=dict(text="CR3BP — 慣性座標系 (ECI · 星空)", font_color="white"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot — Free-Return vs Hohmann overlay (rotating frame)
# ─────────────────────────────────────────────────────────────────────────────

def plot_free_return_compare(fr_history: list, hoh_history: list,
                              lp_dict: Dict[str, np.ndarray],
                              show_soi: bool = True) -> go.Figure:
    """Rotating-frame 3D overlay: free-return (cyan) vs Hohmann reference (orange)."""
    fig = go.Figure()

    for hist, color, label in [
        (fr_history,  "cyan",   "Free-Return (Artemis II 型)"),
        (hoh_history, "orange", "Hohmann 轉移（參考）"),
    ]:
        th = _thin(hist, max_pts=1500)
        fig.add_trace(go.Scatter3d(
            x=[s.pos[0] for s in th],
            y=[s.pos[1] for s in th],
            z=[s.pos[2] for s in th],
            mode="lines", line=dict(color=color, width=3), name=label,
        ))

    xe, ye, ze = _sphere(-MU_STAR, 0, 0, sim.R_EARTH / D_EM)
    fig.add_trace(go.Surface(x=xe, y=ye, z=ze, colorscale="Blues",
                              showscale=False, name="Earth", opacity=0.85,
                              hoverinfo="skip"))
    xm, ym, zm = _sphere(1 - MU_STAR, 0, 0, sim.R_MOON / D_EM)
    fig.add_trace(go.Surface(x=xm, y=ym, z=zm,
                              colorscale=[[0, "#777"], [1, "#ddd"]],
                              showscale=False, name="Moon", opacity=0.85,
                              hoverinfo="skip"))
    if show_soi:
        r_soi = sim.BODIES["Moon"]["soi_km"] / D_EM
        xs_s, ys_s, zs_s = _sphere(1 - MU_STAR, 0, 0, r_soi, n=28)
        fig.add_trace(go.Surface(
            x=xs_s, y=ys_s, z=zs_s,
            colorscale=[[0, "rgba(140,170,255,0.06)"], [1, "rgba(140,170,255,0.06)"]],
            showscale=False, opacity=0.10, hoverinfo="skip", name="月球 SOI",
        ))
    for nm, pos in lp_dict.items():
        if nm in ("L1", "L2"):
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode="markers+text", text=[nm], textposition="top center",
                marker=dict(size=6, color="gold", symbol="diamond"),
                name=nm,
            ))

    sc = {
        "xaxis": dict(**_ENG_SCENE["xaxis"], title="x (norm.)", range=[-1.5, 1.5]),
        "yaxis": dict(**_ENG_SCENE["yaxis"], title="y (norm.)", range=[-1.5, 1.5]),
        "zaxis": dict(**_ENG_SCENE["zaxis"], title="z (norm.)"),
        "bgcolor": _ENG_BG, "aspectmode": "data",
    }
    fig.update_layout(
        **_LAYOUT_BASE, scene=sc, height=600, scene_camera=_CAMERA,
        title=dict(text="Free-Return vs Hohmann — 旋轉座標系對比（20天）", font_color="white"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot — Zero-velocity curves with Lagrange-point thresholds
# ─────────────────────────────────────────────────────────────────────────────

def plot_zvc(jacobi_c: float, jacobi_lp: Optional[Dict[str, float]] = None,
             n: int = 280) -> go.Figure:
    x = np.linspace(-1.55, 1.55, n)
    y = np.linspace(-1.25, 1.25, n)
    X, Y = np.meshgrid(x, y)
    r1 = np.sqrt((X + MU_STAR) ** 2 + Y ** 2 + 1e-30)
    r2 = np.sqrt((X - 1 + MU_STAR) ** 2 + Y ** 2 + 1e-30)
    U  = 0.5 * (X ** 2 + Y ** 2) + (1 - MU_STAR) / r1 + MU_STAR / r2
    Z  = 2 * U - jacobi_c

    z_forbidden = np.where(Z < 0, np.clip(Z, -3, 0), np.nan)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=x, y=y, z=z_forbidden,
        colorscale=[[0, "rgba(60,80,220,0.55)"], [1, "rgba(60,80,220,0.0)"]],
        showscale=False, zmin=-3, zmax=0, hoverinfo="skip",
    ))

    fig.add_trace(go.Contour(
        x=x, y=y, z=Z,
        contours=dict(start=0, end=0.001, size=0.002, coloring="none"),
        line=dict(color="white", width=2.5),
        showscale=False, name=f"ZVC  C={jacobi_c:.5f}",
    ))

    if jacobi_lp:
        for lp_name, color, lbl in [("L1", "orange", "C_L1"), ("L2", "yellow", "C_L2")]:
            c_ref = jacobi_lp.get(lp_name)
            if c_ref is not None:
                Z_ref = 2 * U - c_ref
                fig.add_trace(go.Contour(
                    x=x, y=y, z=Z_ref,
                    contours=dict(start=0, end=0.001, size=0.002, coloring="none"),
                    line=dict(color=color, width=1, dash="dash"),
                    showscale=False, name=lbl,
                ))

    fig.add_trace(go.Scatter(
        x=[-MU_STAR, 1 - MU_STAR], y=[0, 0], mode="markers",
        marker=dict(size=[13, 7], color=["royalblue", "silver"]),
        name="Bodies",
    ))

    lp_all = sim.CR3BPPropagator.lagrange_points()
    for nm, pos in lp_all.items():
        fig.add_trace(go.Scatter(
            x=[pos[0]], y=[pos[1]], mode="markers+text",
            text=[nm], textposition="top center",
            marker=dict(size=7, color="gold", symbol="diamond"),
            showlegend=False,
        ))

    fig.update_layout(
        xaxis=dict(title="x (norm.)", scaleanchor="y", scaleratio=1,
                   range=[-1.55, 1.55], gridcolor=_GRID),
        yaxis=dict(title="y (norm.)", range=[-1.25, 1.25], gridcolor=_GRID),
        title=dict(text=f"零速度曲線 (ZVC)   C = {jacobi_c:.5f}", font_color="white"),
        plot_bgcolor=_DARK, paper_bgcolor=_DARK, font_color="white",
        height=440, margin=dict(l=0, r=0, b=0, t=42),
        legend=dict(bgcolor="rgba(20,24,40,0.85)"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot — Porkchop with contour overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_porkchop(grid: sim.PorkchopGrid,
                  sel_dep: Optional[float] = None,
                  sel_tof: Optional[float] = None,
                  show_contours: bool = True) -> go.Figure:
    dv_ms   = grid.dv_grid.T * 1000.0
    dv_plot = np.where(np.isnan(dv_ms), None, np.clip(dv_ms, 0, 8000))
    dv_cont = np.where(np.isnan(dv_ms), np.nan, dv_ms)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=grid.dep_days, y=grid.tof_days, z=dv_plot,
        colorscale="RdYlGn_r",
        colorbar=dict(title="Total ΔV<br>(m/s)", thickness=14),
        zmin=2000, zmax=7000,
        hovertemplate=(
            "出發 J2000+%{x:.0f}d<br>TOF %{y:.0f}d<br>ΔV %{z:.0f} m/s<extra></extra>"
        ),
    ))

    if show_contours:
        fig.add_trace(go.Contour(
            x=grid.dep_days, y=grid.tof_days, z=dv_cont,
            contours=dict(
                start=2000, end=7000, size=500,
                showlabels=True,
                labelfont=dict(size=9, color="rgba(255,255,255,0.9)"),
                coloring="lines",
            ),
            line=dict(color="rgba(255,255,255,0.50)", width=1),
            showscale=False, name="ΔV 等值線",
            colorscale=[[0, "white"], [1, "white"]],
            hoverinfo="skip",
        ))

    if grid.best_leg:
        fig.add_trace(go.Scatter(
            x=[grid.best_leg.dep_t_s / 86400],
            y=[grid.best_leg.tof_days],
            mode="markers",
            marker=dict(size=14, color="cyan", symbol="star",
                        line=dict(color="black", width=1)),
            name=f"全局最佳 {grid.best_leg.total_dv_kms * 1000:.0f} m/s",
        ))

    if sel_dep is not None and sel_tof is not None:
        fig.add_trace(go.Scatter(
            x=[sel_dep], y=[sel_tof], mode="markers",
            marker=dict(size=12, color="white", symbol="circle-open",
                        line=dict(color="white", width=2)),
            name="選取點",
        ))

    fig.update_layout(
        xaxis_title="出發日 (J2000 天)", yaxis_title="飛行時間 (天)",
        title=dict(text="Earth → Mars 發射窗口掃描 (Porkchop Plot)", font_color="white"),
        paper_bgcolor=_DARK, font_color="white",
        height=440, margin=dict(l=0, r=0, b=0, t=42),
        legend=dict(bgcolor="rgba(20,24,40,0.85)"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot — Hyperbolic arrival inset (2D)
# ─────────────────────────────────────────────────────────────────────────────

def plot_hyperbolic_arrival(v_inf_kms: float, r_cap_km: float,
                             mu: float = MU_MARS) -> go.Figure:
    if v_inf_kms <= 0 or r_cap_km <= 0:
        return go.Figure()

    a     = mu / (v_inf_kms ** 2)
    e     = 1.0 + r_cap_km / a
    p     = a * (e ** 2 - 1.0)
    theta_inf = math.acos(max(-1.0, min(1.0, -1.0 / e)))
    theta     = np.linspace(-theta_inf * 0.91, theta_inf * 0.91, 600)
    r         = p / (1.0 + e * np.cos(theta))
    x_hyp     = r * np.cos(theta)
    y_hyp     = r * np.sin(theta)

    th_c   = np.linspace(0, 2 * math.pi, 200)
    x_circ = r_cap_km * np.cos(th_c)
    y_circ = r_cap_km * np.sin(th_c)
    x_mars = R_MARS * np.cos(th_c)
    y_mars = R_MARS * np.sin(th_c)

    v_peri = math.sqrt(v_inf_kms ** 2 + 2 * mu / r_cap_km)
    v_circ = math.sqrt(mu / r_cap_km)
    dv_loi = v_peri - v_circ

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_mars, y=y_mars, mode="lines",
        fill="toself", fillcolor="rgba(180,60,20,0.55)",
        line=dict(color="#c83c14", width=1), name="Mars",
    ))

    mask_in  = theta <= 0
    mask_out = theta >= 0
    fig.add_trace(go.Scatter(
        x=x_hyp[mask_in], y=y_hyp[mask_in], mode="lines",
        line=dict(color="#ffaa00", width=2.5), name="接近段",
    ))
    fig.add_trace(go.Scatter(
        x=x_hyp[mask_out], y=y_hyp[mask_out], mode="lines",
        line=dict(color="#ffaa00", width=1.5, dash="dot"), name="無捕獲延伸",
    ))
    fig.add_trace(go.Scatter(
        x=x_circ, y=y_circ, mode="lines",
        line=dict(color="cyan", width=1.5, dash="dash"),
        name=f"捕獲軌道  r={r_cap_km:.0f} km",
    ))
    fig.add_trace(go.Scatter(
        x=[r_cap_km], y=[0], mode="markers+text",
        text=[f" ΔV_LOI={dv_loi * 1000:.0f} m/s"],
        textposition="middle right",
        textfont=dict(color="yellow", size=11),
        marker=dict(size=11, color="yellow", symbol="star"), name="近拱點",
    ))
    fig.add_annotation(
        x=float(x_hyp[0]) * 0.55, y=float(y_hyp[0]) * 0.70,
        text=(f"<b>雙曲線捕獲</b><br>v∞ = {v_inf_kms * 1000:.0f} m/s<br>"
              f"e = {e:.3f}<br>v_近 = {v_peri * 1000:.0f} m/s<br>"
              f"v_圓 = {v_circ * 1000:.0f} m/s<br>"
              f"<b>ΔV_LOI = {dv_loi * 1000:.0f} m/s</b>"),
        showarrow=False, font=dict(size=11, color="white"),
        bgcolor="rgba(20,20,40,0.88)",
        bordercolor="rgba(150,150,200,0.7)", borderwidth=1, align="left",
    )

    r_max = max(r_cap_km * 5.5, float(np.abs(x_hyp).max()) * 1.05)
    fig.update_layout(
        xaxis=dict(title="x (km)", scaleanchor="y", scaleratio=1,
                   range=[-r_max, r_max], gridcolor=_GRID, showgrid=True),
        yaxis=dict(title="y (km)", range=[-r_max * 0.6, r_max * 0.6],
                   gridcolor=_GRID, showgrid=True),
        plot_bgcolor=_DARK, paper_bgcolor=_DARK, font_color="white",
        title=dict(text="火星雙曲線捕獲軌道 (LOI 插圖)", font_color="white"),
        height=360, margin=dict(l=0, r=0, b=0, t=42),
        legend=dict(bgcolor="rgba(20,24,40,0.85)", font_size=10, x=0.01, y=0.99),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — mode selector is always at the top
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛰️ Open Mission Ops Lab")

    mode = st.radio(
        "介面模式",
        ["🎓 教學模式 (Step-by-step)", "🛠 專家模式"],
        horizontal=False,
    )
    st.divider()

    # ── Teaching mode sidebar ─────────────────────────────────────────────────
    if mode.startswith("🎓"):
        edu_step = st.session_state.get("edu_step", 0)

        # Step progress indicator
        _step_names = ["1. CR3BP", "2. Jacobi/ZVC", "3. ΔV 計算"]
        st.caption(f"**目前步驟：{_step_names[edu_step]}**")
        st.progress(edu_step / 2.0)

        st.divider()
        st.subheader("模擬參數")
        st.caption("参數會隨步驟逐步解鎖。")

        alt_leo = int(st.number_input(
            "停泊軌道高度 (km)", 200, 800, 200, step=50, key="edu_leo"))
        _v_hoh = _hohmann_v(alt_leo)

        if edu_step >= 1:
            v_boost = float(st.slider(
                "TLI 速度 (km/s)", 3.05, 3.30,
                float(round(_v_hoh, 3)), step=0.002, key="edu_v",
                help=f"Hohmann 最佳值 ≈ {_v_hoh:.3f} km/s",
            ))
        else:
            v_boost = _v_hoh
            st.caption(f"TLI 速度固定為 Hohmann 最佳值：**{_v_hoh:.3f} km/s**"
                       "\n（步驟 2 解鎖調整）")

        if edu_step >= 2:
            alt_lmo = int(st.number_input(
                "月球目標軌道高度 (km)", 50, 500, 100, step=50, key="edu_lmo"))
        else:
            alt_lmo = 100

        sim_days = 5.0
        n_steps  = 5000
        run_btn  = False

    # ── Expert mode sidebar ───────────────────────────────────────────────────
    else:
        edu_step = -1
        st.caption("CR3BP / 地月 / 行星際 任務分析平台")
        st.divider()

        mission = st.selectbox("任務模式",
                                ["🌕 地月轉移 (Cislunar)", "🔴 行星際 (Earth → Mars)"])
        st.divider()

        if mission.startswith("🌕"):
            st.subheader("軌道初始條件")
            alt_leo = int(st.number_input("停泊軌道高度 (km)", 200, 1000, 200, step=50))
            alt_lmo = int(st.number_input("月球目標軌道高度 (km)", 50, 500, 100, step=50))
            _v_hoh  = _hohmann_v(alt_leo)
            v_boost = float(st.slider(
                "TLI 速度 (km/s)", 3.05, 3.30,
                float(round(_v_hoh, 3)), step=0.002,
                help=f"Hohmann TLI 速度 ≈ {_v_hoh:.3f} km/s",
            ))
            sim_days = float(st.slider("積分時間 (天)", 1.0, 14.0, 3.5, step=0.5))
            n_steps  = int(st.select_slider("積分步數（精度）",
                                             [1000, 2000, 5000, 10000], value=5000))
            st.divider()
            st.subheader("顯示選項")
            frame_mode   = st.radio("坐標系視圖",
                                     ["旋轉系 (Rotating)", "慣性系 (Inertial)"])
            show_zvc     = st.checkbox("顯示零速度曲線 (ZVC)", value=True)
            show_soi     = st.checkbox("顯示月球 SOI 球體", value=True)

        else:
            st.subheader("掃描視窗")
            dep_start     = int(st.number_input("出發日 起始 (J2000 天)", value=9690, step=10))
            dep_end       = int(st.number_input("出發日 截止 (J2000 天)", value=9950, step=10))
            tof_min       = int(st.number_input("TOF 最短 (天)", value=130, step=10))
            tof_max       = int(st.number_input("TOF 最長 (天)", value=320, step=10))
            _res_map      = {"快速 (20×15)": (20, 15),
                             "中等 (40×30)": (40, 30),
                             "精細 (80×60)": (80, 60)}
            res_key       = st.select_slider("掃描解析度", list(_res_map.keys()),
                                              value="中等 (40×30)")
            dep_n, tof_n  = _res_map[res_key]
            show_contours = st.checkbox("顯示 ΔV 等值線", value=True)

        st.divider()
        run_btn = st.button("▶ 執行計算", type="primary", use_container_width=True)

    # ── Mission Log (shared) ──────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 任務記錄")
    log = st.session_state["mission_log"]
    if log:
        for i, entry in enumerate(reversed(log)):
            idx = len(log) - 1 - i
            with st.expander(f"{entry['label'][:28]}… — {entry['dv_total_ms']:.0f} m/s"):
                st.caption(entry["details"])
                if st.button("🗑 刪除", key=f"del_{idx}"):
                    st.session_state["mission_log"].pop(idx)
                    st.rerun()
    else:
        st.caption("尚無記錄。執行計算後點選「💾 儲存當前解」。")


# ═════════════════════════════════════════════════════════════════════════════
# TEACHING MODE — 3-step wizard
# ═════════════════════════════════════════════════════════════════════════════
if mode.startswith("🎓"):

    SS = st.session_state

    # Run simulation (cached; auto-triggered by parameter changes)
    with st.spinner("模擬中…"):
        history, lp = run_cr3bp(alt_leo, round(v_boost, 4), sim_days, n_steps)
        plan         = run_plan(alt_leo, alt_lmo)
    jlp = get_lagrange_jacobi()

    # Step tracker (circles at top)
    _ic = ["🔵", "⭕", "⭕"]
    _ic[edu_step] = "🔵"
    for i in range(edu_step):
        _ic[i] = "✅"

    _s1c, _s2c, _s3c = st.columns(3)
    _s1c.markdown(f"{_ic[0]} **步驟 1**\nCR3BP 基礎")
    _s2c.markdown(f"{_ic[1]} **步驟 2**\nJacobi / ZVC")
    _s3c.markdown(f"{_ic[2]} **步驟 3**\nΔV 計算")
    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — What is CR3BP?
    # ─────────────────────────────────────────────────────────────────────────
    if edu_step == 0:
        st.markdown("""
<div class="edu-banner">
<strong>步驟 1 / 3：CR3BP 與旋轉座標系</strong> — 理解為什麼需要兩種視角來看地月軌道
</div>""", unsafe_allow_html=True)

        with st.expander("📖 什麼是 CR3BP？（點開閱讀）", expanded=True):
            _tc1, _tc2 = st.columns(2)
            with _tc1:
                st.markdown("""
**圓形受限三體問題（Circular Restricted Three-Body Problem, CR3BP）**

假設地球與月球沿固定圓軌道互繞，第三個質量可忽略的太空船只受這兩個天體的引力影響。

**什麼是 Barycenter（質心）？**

地球和月球都繞著它們的共同質心（barycenter）公轉。由於地球質量遠大於月球，這個質心位於地球內部，距地心約 **4,670 km**。

| | 慣性座標系 (ECI) | 旋轉座標系 |
|--|--|--|
| 地球位置 | 近似固定 | 固定在 (−μ*, 0) |
| 月球位置 | 沿圓軌道運動 | 固定在 (1−μ*, 0) |
| 軌跡外觀 | 螺旋或橢圓 | 直觀、對稱 |
| 新增力項 | 無 | 離心力 + 科氏力 |
""")
            with _tc2:
                st.markdown("""
**為什麼要用旋轉座標系？**

讓座標系以地月公轉角速度 n* 旋轉後，地球與月球在座標中「靜止不動」，成為固定引力源。

好處：
- 軌道圖直接顯示太空船相對於地月系統的軌跡
- L1～L5 拉格朗日點在旋轉系中是固定的平衡點
- 暈軌道（Halo Orbit）在旋轉系中看起來是封閉曲線

缺點：旋轉座標系引入**科氏力**與**離心力**（非慣性力），需要在運動方程式裡加以處理。

本程式使用的旋轉系歸一化單位：
- 長度：地月距離 $D_{EM}$ ≈ 384,400 km ≡ 1
- 時間：月球公轉週期 / (2π) ≡ 1
- 地球位於 (−μ*, 0)，月球位於 (1−μ*, 0)
""")

        st.subheader("並排比較：相同軌道在兩種座標系的外觀")
        _pc1, _pc2 = st.columns(2)
        with _pc1:
            st.caption("**旋轉座標系**（地球、月球固定，軌跡呈現 CR3BP 特徵）")
            st.plotly_chart(
                plot_rotating(history, lp, scrub_idx=None, show_soi=False),
                use_container_width=True,
            )
        with _pc2:
            st.caption("**慣性座標系 (ECI)**（月球沿虛線圓弧移動）")
            st.plotly_chart(plot_inertial(history, scrub_idx=None),
                            use_container_width=True)

        st.info(
            "**觀察**：左圖（旋轉系）中，軌跡在月球附近「彎曲」——這是 L1 點附近的三體動力學效應。"
            "右圖（慣性系）中，同一條軌道看起來像標準的轉移橢圓，月球沿虛線弧移動。"
        )

        st.divider()
        _nav_buttons(fwd_label="➡ 下一步：Jacobi 常數與零速度曲線", fwd_step=1)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — Jacobi constant & ZVC
    # ─────────────────────────────────────────────────────────────────────────
    elif edu_step == 1:
        st.markdown("""
<div class="edu-banner">
<strong>步驟 2 / 3：Jacobi 常數與零速度曲線（ZVC）</strong> — 旋轉系中的能量守恆
</div>""", unsafe_allow_html=True)

        with st.expander("📖 什麼是 Jacobi 常數？", expanded=True):
            st.markdown("""
在旋轉座標系裡，有一個特殊的量保持不變，稱為 **Jacobi 常數 C**（又叫 Jacobi 積分）：
""")
            st.latex(r"C \;=\; 2U(x,\,y,\,z) \;-\; v^{2}")
            st.markdown(r"""
- $U = \frac{1}{2}(x^2+y^2) + \frac{1-\mu^*}{r_1} + \frac{\mu^*}{r_2}$ 是**有效位能**（含離心項）
- $v$ 是在旋轉座標系裡量測的速度大小
- 在普通的慣性系裡有「總能量守恆」；在旋轉系裡則是 Jacobi 常數守恆

**零速度曲線（Zero-Velocity Curve, ZVC）**：令 $v = 0$，得到 $C = 2U$，這條面稱為零速度曲面。

| 區域 | 意義 |
|------|------|
| 藍色陰影（禁止區域） | 此處 $2U < C$，動能為負，太空船進不去 |
| 白色邊界線 | 當前能量的 ZVC（$2U = C$）|
| 橙色虛線 | $C_{L1}$ 門檻：剛好在 L1 點打開缺口的能量 |
| 黃色虛線 | $C_{L2}$ 門檻：L2 點對應的能量 |
""")

        # Use midpoint of trajectory for scrubber
        _scrub = len(history) // 2
        c_cur = history[_scrub].jacobi()
        c_l1  = jlp.get("L1", float("nan"))
        c_l2  = jlp.get("L2", float("nan"))

        _zc1, _zi1 = st.columns([3, 1])
        with _zc1:
            st.plotly_chart(plot_zvc(c_cur, jacobi_lp=jlp), use_container_width=True)
        with _zi1:
            if c_cur > c_l1:
                _status_txt   = "🔴 L1 通道關閉\n太空船被鎖在地球附近"
                _status_color = "#ff4b4b"
            elif c_cur > c_l2:
                _status_txt   = "🟡 L1 通道開啟\n能量足以到達月球附近"
                _status_color = "#ffcc00"
            else:
                _status_txt   = "🟢 L1 + L2 均開啟\n可逃逸到月球外側"
                _status_color = "#00cc66"

            st.markdown(f"""
**當前 Jacobi 常數**

`C = {c_cur:.5f}`

| 閾值 | 數值 |
|------|------|
| $C_{{L1}}$ | `{c_l1:.5f}` |
| $C_{{L2}}$ | `{c_l2:.5f}` |
| 當前 C | `{c_cur:.5f}` |

**通道狀態：**
<span style="color:{_status_color};">{_status_txt.replace(chr(10), "<br>")}</span>
""", unsafe_allow_html=True)

        st.info(
            "**互動任務**：在左側邊欄拉動「TLI 速度」滑桿（已解鎖），觀察 Jacobi C 如何變化。"
            " 速度越低 → C 越大 → ZVC 白色邊界關閉 L1 通道；"
            " 速度越高 → C 越小 → 橙色 C_L1 曲線通道打開，可到達月球區域。"
        )

        st.markdown(f"""
**Jacobi 常數物理意義速查**

| 條件 | 物理意義 |
|------|---------|
| $C > C_{{L1}} = {c_l1:.4f}$ | 太空船無法穿越 L1，被困在地球附近 |
| $C_{{L2}} < C \\leq C_{{L1}}$ | L1 通道打開，可以前往月球；L2 仍封閉 |
| $C \\leq C_{{L2}} = {c_l2:.4f}$ | L1 和 L2 均開啟，可以進入外側空間 |
""")

        st.divider()
        _nav_buttons(
            back_label="⬅ 回到步驟 1", back_step=0,
            fwd_label="➡ 下一步：地月任務 ΔV 計算", fwd_step=2,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — TLI / LOI ΔV
    # ─────────────────────────────────────────────────────────────────────────
    elif edu_step == 2:
        st.markdown("""
<div class="edu-banner">
<strong>步驟 3 / 3：地月任務 ΔV — TLI 與 LOI（Patched Conics）</strong>
</div>""", unsafe_allow_html=True)

        st.markdown("""
我們用 **Patched Conics（拼接圓錐曲線）** 近似，把整個任務分成兩段獨立的二體問題，分別計算需要的速度增量（ΔV）：

1. **TLI（Trans-Lunar Injection，月球轉移推進）**：在地球停泊軌道點火，進入一條遠地點在月球距離的轉移橢圓。
2. **LOI（Lunar Orbit Insertion，月球軌道插入）**：在月球附近減速，從雙曲線飛越捕獲為環月圓軌道。
""")

        # Pre-compute values for LaTeX
        r_leo     = sim.R_EARTH + alt_leo
        a_tl      = 0.5 * (r_leo + sim.D_EM)
        v_leo_val = math.sqrt(sim.MU_EARTH / r_leo)
        v_tli_val = math.sqrt(sim.MU_EARTH * (2.0 / r_leo - 1.0 / a_tl))
        r_lmo     = sim.R_MOON + alt_lmo
        v_lmo_val = math.sqrt(sim.MU_MOON / r_lmo)

        _m1, _m2 = st.columns([1, 1])

        with _m1:
            st.subheader("ΔV 預算一覽")
            st.metric("ΔV_TLI", f"{plan.dv_tli_kms * 1000:.1f} m/s",
                      help="從地球停泊軌道到月球轉移軌道的點火量")
            st.metric("ΔV_LOI", f"{plan.dv_loi_kms * 1000:.1f} m/s",
                      help="月球附近的制動減速量")
            st.metric("合計 ΔV",
                      f"{(plan.dv_tli_kms + plan.dv_loi_kms) * 1000:.1f} m/s")
            st.divider()
            st.metric("飛行時間 (TOF)",   f"{plan.tof_days:.2f} 天")
            st.metric("月球 SOI 超越速度 v∞",
                      f"{plan.v_inf_moon_kms * 1000:.0f} m/s")
            st.metric("Jacobi 近似值 C",  f"{plan.jacobi_approx:.4f}")
            st.divider()
            st.info("試著在側邊欄改變 LEO 高度或月球目標軌道高度，觀察 ΔV_TLI 與 ΔV_LOI 如何改變。")
            if st.button("💾 儲存當前解", use_container_width=True):
                st.session_state["mission_log"].append({
                    "label": f"地月 LEO={alt_leo}km→LMO={alt_lmo}km (教學)",
                    "dv_total_ms": (plan.dv_tli_kms + plan.dv_loi_kms) * 1000,
                    "details": (
                        f"TLI={plan.dv_tli_kms * 1000:.1f} m/s, "
                        f"LOI={plan.dv_loi_kms * 1000:.1f} m/s, "
                        f"TOF={plan.tof_days:.2f}d"
                    ),
                })
                st.success("已儲存至任務記錄！")

        with _m2:
            st.subheader("物理方程式推導（附計算結果）")

            st.markdown("**① 轉移橢圓半長軸**：起點軌道半徑與終點（月球距離）的算術平均：")
            st.latex(r"a_{TL} = \frac{r_{LEO} + D_{EM}}{2}")
            st.latex(
                rf"= \frac{{{r_leo:.0f} + {sim.D_EM:.0f}}}{{2}}"
                rf"= \mathbf{{{a_tl:.0f}}} \text{{ km}}"
            )

            st.markdown("**② TLI 速度差**：用 vis-viva 方程計算轉移橢圓近地點速度，減去圓軌道速度：")
            st.latex(
                r"\Delta V_{TLI} = \sqrt{\mu_E\!\left(\frac{2}{r_{LEO}}-\frac{1}{a_{TL}}\right)}"
                r"- \sqrt{\frac{\mu_E}{r_{LEO}}}"
            )
            st.latex(
                rf"= {v_tli_val:.4f} - {v_leo_val:.4f}"
                rf"= \mathbf{{{plan.dv_tli_kms * 1000:.1f}}} \text{{ m/s}}"
            )

            st.markdown(
                "**③ LOI 速度差**：月球 SOI 入口超越速度 $v_\\infty$ 與圓軌道速度合成後，"
                "減去目標圓軌道速度：")
            st.latex(r"\Delta V_{LOI} = \sqrt{v_{c,LMO}^2 + v_\infty^2} - v_{c,LMO}")
            st.latex(
                rf"= \sqrt{{{v_lmo_val:.4f}^2 + {plan.v_inf_moon_kms:.4f}^2}}"
                rf"- {v_lmo_val:.4f}"
                rf"= \mathbf{{{plan.dv_loi_kms * 1000:.1f}}} \text{{ m/s}}"
            )

            c_approx = plan.jacobi_approx
            c_l1     = jlp.get("L1", float("nan"))
            st.markdown(
                f"**Jacobi 常數驗證**：Patched Conics 近似值 C ≈ {c_approx:.4f}，"
                f"CR3BP 積分起點值 C₀ = {history[0].jacobi():.4f}。"
                f"兩者差異源自 Patched Conics 在 SOI 邊界的不連續假設。"
            )

        # ── Artemis II: why not Hohmann?
        st.divider()
        with st.expander("🚀 Artemis II 為什麼不用 Hohmann？", expanded=True):
            st.markdown("""
**Artemis II（2026 年 4 月，首次載人深空任務）使用 Free-Return Trajectory，而非 Hohmann 轉移。**

| 項目 | 本程式（Hohmann 近似） | Artemis II 實際 |
|------|----------------------|----------------|
| 軌道形狀 | 簡單橢圓（單程） | Figure-8 自由返回（往返） |
| 月球模型 | 固定在遠地點 | 移動（真實星曆） |
| 引力模型 | 地心二體 + SOI patch | CR3BP / 完整三體 |
| ΔV 總量 | ~4,000 m/s（TLI + LOI） | ~3,200 m/s（僅 TLI） |
| 安全性 | 需要 LOI 才能返回 | 被動安全，引擎故障可自動返回 |
| 飛行時間 | ~3.5 天單程 | ~10 天往返 |
""")
            _why1, _why2 = st.columns(2)
            with _why1:
                st.markdown("""
**原因一：載人安全優先**

Free-return 軌道的最大優勢是**被動安全性（Passive Safety）**。

Apollo 13 危機（1970 年）：服務艙引擎爆炸損壞，正是靠 free-return 軌道讓太空船**不需要任何推進點火**，靠重力自動返回地球。

Artemis II 作為 50 年來首次載人深空任務，必須優先考慮**失效模式保護**——一旦推進系統出問題，太空船靠三體重力自然飛回地球，零風險。

**Jacobi 常數視角**：軌道能量被精確設計使 $C < C_{L1}$，讓軌道穿越 L1 通道到月球，繞過月球後月球重力「拋出」，沿 figure-8 返回地球。
""")
            with _why2:
                st.markdown("""
**原因二：任務目標不同（飛掠驗證）**

Artemis II 是**飛掠測試任務**，不需要進入月球軌道（LOI），因此不需 Hohmann 轉移：

| 驗證項目 | 細節 |
|---------|------|
| 生命維持 | Orion 太空船 10 天連續運作 |
| 深空通訊 | 地月 40 萬 km 通訊穩定性 |
| 手動操控 | 模擬對接演練 (rendezvous) |
| 輻射暴露 | 深空輻射劑量測量 |

**實際飛行數據（2026 年 4 月）**
- 最遠距離：406,700 km（超越 Apollo 13 紀錄）
- 近月距離：**6,513 km**（月球中心）
- TLI 速度：10.8–11.2 km/s（略高於 Hohmann）
- 不進入月球軌道 → 不需 LOI → 節省 ~800 m/s ΔV
""")

        # ── Free-Return live simulation
        st.divider()
        st.subheader("🔄 Free-Return 軌道模擬")
        st.caption(
            "程式使用**二分法（Bisection）**搜尋：從 Hohmann TLI 速度出發，"
            "反覆調整 $v_{TLI}$，直到近月距離符合目標值。"
            "首次執行約需 20–40 秒（結果快取，再次點擊瞬間）。"
        )

        _fr_peri = int(st.number_input(
            "目標近月距離 (km，月球中心)", 2000, 20000, 6500, step=500,
            key="fr_peri_edu",
            help="Artemis II 實際值 = 6,513 km | 月球半徑 = 1,737 km | 月球 SOI = 66,100 km",
        ))
        if st.button("🚀 執行 Free-Return 二分法搜尋", type="primary",
                     use_container_width=True, key="run_fr_edu"):
            st.session_state["fr_edu_computed"] = True

        if st.session_state.get("fr_edu_computed"):
            with st.spinner("二分法搜尋 Free-Return 軌道…（首次約 60–120 秒，結果快取）"):
                _fr_hist, _fr_v, _fr_peri_actual = run_free_return(alt_leo, _fr_peri)
            _v_hoh_fr = _hohmann_v(alt_leo)
            _c_fr = _fr_hist[0].jacobi()
            _c_l1_fr = jlp.get("L1", float("nan"))
            _c_l2_fr = jlp.get("L2", float("nan"))

            _fc1, _fc2, _fc3, _fc4 = st.columns(4)
            _fc1.metric("TLI 速度（找到）", f"{_fr_v:.4f} km/s",
                        delta=f"+{(_fr_v - _v_hoh_fr) * 1000:.1f} m/s vs Hohmann")
            _fc2.metric("實際近月距離", f"{_fr_peri_actual:.0f} km",
                        delta=f"目標 {_fr_peri} km")
            _fc3.metric("Jacobi C (TLI 起點)", f"{_c_fr:.5f}")
            _fc4.metric("C_L1 門檻", f"{_c_l1_fr:.5f}",
                        delta="C < C_L1 ✅" if _c_fr < _c_l1_fr else "C ≥ C_L1 ⚠",
                        delta_color="normal" if _c_fr < _c_l1_fr else "inverse")

            if _c_fr < _c_l2_fr:
                st.success(
                    f"L1 + L2 均開啟 (C={_c_fr:.5f} < C_L2={_c_l2_fr:.5f})："
                    "能量足以穿越月球兩側，形成 figure-8 free-return 軌道。"
                )
            elif _c_fr < _c_l1_fr:
                st.info(
                    f"L1 通道開啟 (C_L2={_c_l2_fr:.5f} < C={_c_fr:.5f} < C_L1={_c_l1_fr:.5f})："
                    "可穿越 L1 到達月球區域，靠月球引力形成 free-return。"
                )
            else:
                st.warning(
                    f"C={_c_fr:.5f} ≥ C_L1={_c_l1_fr:.5f}：L1 通道關閉，"
                    "此速度無法到達月球。請增大近月距離目標值（降低所需 v_TLI）。"
                )

            _hoh_10_hist = run_hohmann_cr3bp(alt_leo)
            st.plotly_chart(
                plot_free_return_compare(_fr_hist, _hoh_10_hist, lp),
                use_container_width=True,
            )
            st.caption(
                "**青色線** = Free-Return 軌道（20天，含完整 figure-8 返回段）；"
                "**橙色線** = Hohmann 轉移（20天漂移，需 LOI 才能返回）。"
                "  旋轉坐標系中，地球（左，x≈−0.012）與月球（右，x≈0.988）固定不動。"
                "  近月點在月球附近 x 軸上（L1 側），返回段經過 L4/L5 下方區域後返回地球。"
                "  ⚠️ 此為二維共面 CR3BP 簡化模型；Artemis II 實際為 28° 傾角三維軌道。"
            )
            st.info(
                "**觀察 figure-8 形狀**：\n"
                "- **去程**（t=−10 至 −2.8d）：青色線從地球出發，弧形向上，"
                "  在月球附近達到近月點（x≈0.97，幾乎在 L1–L2 連線上）。\n"
                "- **返程**（t=−2.8 至 +6.5d）：月球重力將太空船「拋出」，"
                "  軌跡向下弧形（穿越 L5 附近區域），約 16.5 天後回到地球（近地距離≈15,000 km）。\n"
                "- **Hohmann（橙色）**：到達月球距離後繼續漂移遠離，沒有自然返回。"
            )

        st.divider()
        st.success(
            "🎓 完成教學導覽！你已學習：\n\n"
            "1. CR3BP 的旋轉座標系與慣性座標系差異\n"
            "2. Jacobi 常數與 ZVC 能量通道\n"
            "3. Patched Conics 的 TLI / LOI ΔV 計算\n"
            "4. Free-Return 軌道的被動安全設計與 Artemis II 任務邏輯\n\n"
            "切換左側邊欄到「🛠 專家模式」可使用完整的 3D 軌道視覺化、"
            "時間軸 scrubber、CZML 匯出，以及 Earth→Mars 行星際分析。"
        )
        _nav_buttons(back_label="⬅ 回到步驟 2", back_step=1)


# ═════════════════════════════════════════════════════════════════════════════
# EXPERT MODE — Cislunar CR3BP view
# ═════════════════════════════════════════════════════════════════════════════
elif mission.startswith("🌕"):
    st.header("地月轉移任務 — CR3BP 動力學分析")

    SS = st.session_state
    need_run = run_btn or "cl_history" not in SS

    if need_run:
        with st.status("執行 CR3BP 積分與任務規劃…", expanded=True) as _s:
            st.write("數值積分軌道歷史…")
            history, lp = run_cr3bp(alt_leo, v_boost, sim_days, n_steps)
            st.write("計算 Patched Conics ΔV 預算…")
            plan = run_plan(alt_leo, alt_lmo)
            _s.update(label="計算完成 ✓", state="complete", expanded=False)
        SS["cl_history"] = history
        SS["cl_lp"]      = lp
        SS["cl_plan"]    = plan
    else:
        history = SS["cl_history"]
        lp      = SS["cl_lp"]
        plan    = SS["cl_plan"]

    jlp = get_lagrange_jacobi()

    # Jacobi monitor
    j0    = history[0].jacobi()
    jf    = history[-1].jacobi()
    drift = abs(jf - j0) / abs(j0) * 100
    d_moon = min(s.dist_to_moon_km() for s in history)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Jacobi C₀",  f"{j0:.6f}")
    c2.metric("Jacobi C_f", f"{jf:.6f}",
              delta=f"{jf - j0:+.2e}", delta_color="inverse")
    c3.metric("漂移量",      f"{drift:.3e} %",
              delta="⚠ 增加步數" if drift > 0.01 else "積分良好",
              delta_color="inverse" if drift > 0.01 else "off")
    c4.metric("最近月距",    f"{d_moon:.0f} km")
    c5.metric("終態速度",    f"{history[-1].speed_kms:.3f} km/s")
    st.divider()

    tab3d, tab_zvc, tab_plan, tab_fr = st.tabs(
        ["🌌 3D 軌道圖", "🔵 零速度曲線 (ZVC)", "📋 任務 ΔV 規劃", "🔄 Free-Return 設計"])

    with tab3d:
        scrub_idx = st.slider(
            "時間軸 (time scrubber)", 0, len(history) - 1,
            len(history) // 2, key="scrub",
        )
        s_cur = history[scrub_idx]
        st.caption(
            f"**t = {s_cur.t_days:.3f} 天**  |  "
            f"距月球 {s_cur.dist_to_moon_km():.0f} km  |  "
            f"距地球 {s_cur.dist_to_earth_km():.0f} km  |  "
            f"速度 {s_cur.speed_kms:.3f} km/s  |  "
            f"Jacobi C = {s_cur.jacobi():.6f}"
        )
        if s_cur.dist_to_moon_km() < sim.R_MOON:
            st.error("💥 太空船進入月球內部（撞擊）")
        elif s_cur.dist_to_moon_km() < sim.BODIES["Moon"]["soi_km"]:
            st.success("🌕 太空船在月球 SOI 內")

        if frame_mode.startswith("旋"):
            st.plotly_chart(
                plot_rotating(history, lp, scrub_idx, show_soi=show_soi),
                use_container_width=True,
            )
        else:
            st.plotly_chart(plot_inertial(history, scrub_idx),
                            use_container_width=True)

    with tab_zvc:
        c_zvc, c_info = st.columns([3, 1])
        with c_zvc:
            if show_zvc:
                st.plotly_chart(
                    plot_zvc(history[scrub_idx].jacobi(), jacobi_lp=jlp),
                    use_container_width=True,
                )
            else:
                st.info("已關閉 ZVC（側邊欄切換）。")

        with c_info:
            c_val = history[scrub_idx].jacobi()
            c_l1  = jlp.get("L1", float("nan"))
            c_l2  = jlp.get("L2", float("nan"))
            if c_val > c_l1:
                _stxt, _scol = "🔴 鎖定於地球附近（L1 通道關閉）", "#ff4b4b"
            elif c_val > c_l2:
                _stxt, _scol = "🟡 L1 通道開啟（可抵達月球）", "#ffcc00"
            else:
                _stxt, _scol = "🟢 L1 + L2 均開啟（可逃逸）", "#00cc66"

            st.markdown(f"""
**Jacobi 常數** `C = {c_val:.5f}`

| 閾值 | 數值 |
|------|------|
| $C_{{L1}}$ | `{c_l1:.5f}` |
| $C_{{L2}}$ | `{c_l2:.5f}` |
| 當前 C | `{c_val:.5f}` |

**通道狀態：**
<span style="color:{_scol}">{_stxt}</span>

橙色虛線 = C_L1；黃色虛線 = C_L2 參考邊界。
""", unsafe_allow_html=True)

    with tab_plan:
        cp1, cp2 = st.columns([1, 1])
        with cp1:
            st.subheader("Patched Conics ΔV 預算")
            st.metric("ΔV_TLI",        f"{plan.dv_tli_kms * 1000:.1f} m/s")
            st.metric("ΔV_LOI",        f"{plan.dv_loi_kms * 1000:.1f} m/s")
            st.metric("合計 ΔV",        f"{(plan.dv_tli_kms + plan.dv_loi_kms) * 1000:.1f} m/s")
            st.divider()
            st.metric("轉移橢圓半長軸", f"{plan.a_transfer_km / 1000:.0f} Mm")
            st.metric("飛行時間",       f"{plan.tof_days:.2f} 天")
            st.metric("月球 SOI v∞",    f"{plan.v_inf_moon_kms * 1000:.0f} m/s")
            st.metric("Jacobi C (TLI)", f"{plan.jacobi_approx:.4f}")
            st.divider()
            if st.button("💾 儲存當前解 (地月)", use_container_width=True):
                st.session_state["mission_log"].append({
                    "label": f"地月 LEO={alt_leo}km→LMO={alt_lmo}km v={v_boost:.3f}",
                    "dv_total_ms": (plan.dv_tli_kms + plan.dv_loi_kms) * 1000,
                    "details": (
                        f"TLI={plan.dv_tli_kms * 1000:.1f} m/s, "
                        f"LOI={plan.dv_loi_kms * 1000:.1f} m/s, "
                        f"TOF={plan.tof_days:.2f}d, C={plan.jacobi_approx:.4f}"
                    ),
                })
                st.success("已儲存至任務記錄！")
            czml_bytes = build_czml_cislunar(history)
            st.download_button(
                "⬇ 匯出 CZML (Cesium)", czml_bytes,
                "cislunar_trajectory.czml", "application/json",
                use_container_width=True,
            )

        with cp2:
            st.subheader("物理方程式（含計算值）")
            r_leo     = sim.R_EARTH + alt_leo
            a_tl      = 0.5 * (r_leo + sim.D_EM)
            v_leo_val = math.sqrt(sim.MU_EARTH / r_leo)
            v_tli_val = math.sqrt(sim.MU_EARTH * (2.0 / r_leo - 1.0 / a_tl))
            r_lmo     = sim.R_MOON + alt_lmo
            v_lmo_val = math.sqrt(sim.MU_MOON / r_lmo)

            st.latex(r"a_{TL} = \frac{r_{LEO} + D_{EM}}{2}")
            st.latex(
                rf"= \frac{{{r_leo:.0f} + {sim.D_EM:.0f}}}{{2}}"
                rf"= \mathbf{{{a_tl:.0f}}} \text{{ km}}"
            )
            st.latex(
                r"\Delta V_{TLI} = v_{TLI} - v_{LEO}"
                rf"= {v_tli_val:.4f} - {v_leo_val:.4f}"
                rf"= \mathbf{{{plan.dv_tli_kms * 1000:.1f}}} \text{{ m/s}}"
            )
            st.latex(r"\Delta V_{LOI} = \sqrt{v_{c,LMO}^2 + v_\infty^2} - v_{c,LMO}")
            st.latex(
                rf"= \sqrt{{{v_lmo_val:.4f}^2 + {plan.v_inf_moon_kms:.4f}^2}}"
                rf"- {v_lmo_val:.4f}"
                rf"= \mathbf{{{plan.dv_loi_kms * 1000:.1f}}} \text{{ m/s}}"
            )

    with tab_fr:
        st.subheader("Free-Return 軌道設計")
        st.markdown("""
**Free-Return Trajectory**：以略高於 Hohmann 的 TLI 速度出發，靠月球重力「拋出」，自然飛回地球。
無需 LOI，亦無需返航推進——這是 Apollo 8/10/11 及 Artemis II 選用此軌道的根本原因。

> **Artemis II（2026 年 4 月）** 近月距離 6,513 km，飛行 10 天，節省 ~800 m/s ΔV 且具備被動安全性。
""")
        _fr2_col1, _fr2_col2 = st.columns([1, 2])
        with _fr2_col1:
            _fr2_peri = int(st.number_input(
                "目標近月距離 (km)", 2000, 20000, 6500, step=500,
                key="fr_peri_exp",
                help="Artemis II = 6,513 km | 月球 SOI = 66,100 km",
            ))
            st.caption(
                f"Hohmann TLI: **{_hohmann_v(alt_leo):.4f} km/s**\n\n"
                "Free-Return TLI 通常高出 Hohmann 約 50–200 m/s。"
            )
            _fr2_run = st.button(
                "🚀 執行二分法搜尋", type="primary",
                use_container_width=True, key="run_fr_exp",
            )
            if _fr2_run:
                st.session_state["fr_exp_computed"] = True

        with _fr2_col2:
            if st.session_state.get("fr_exp_computed"):
                with st.spinner("搜尋 Free-Return 軌道…（首次約 60–120 秒，結果快取）"):
                    _fr2_hist, _fr2_v, _fr2_peri_act = run_free_return(alt_leo, _fr2_peri)
                _fr2_v_hoh = _hohmann_v(alt_leo)
                _fr2_c     = _fr2_hist[0].jacobi()
                _fr2_cl1   = jlp.get("L1", float("nan"))
                _fr2_cl2   = jlp.get("L2", float("nan"))

                fa, fb, fc, fd = st.columns(4)
                fa.metric("TLI 速度", f"{_fr2_v:.4f} km/s",
                          delta=f"+{(_fr2_v - _fr2_v_hoh)*1000:.1f} m/s vs Hohmann")
                fb.metric("近月距離", f"{_fr2_peri_act:.0f} km",
                          delta=f"目標 {_fr2_peri} km")
                fc.metric("Jacobi C", f"{_fr2_c:.5f}")
                fd.metric("C_L1", f"{_fr2_cl1:.5f}",
                          delta="C < C_L1 ✅" if _fr2_c < _fr2_cl1 else "C ≥ C_L1 ⚠",
                          delta_color="normal" if _fr2_c < _fr2_cl1 else "inverse")

                if _fr2_c < _fr2_cl2:
                    st.success(f"C={_fr2_c:.5f} < C_L2={_fr2_cl2:.5f}：L1+L2 均開啟，figure-8 free-return 可行。")
                elif _fr2_c < _fr2_cl1:
                    st.info(f"C_L2 < C={_fr2_c:.5f} < C_L1：L1 通道開啟，月球繞行後可返回。")
                else:
                    st.warning("C ≥ C_L1：L1 通道關閉，此速度無法到達月球區域。")

                _hoh_ref_hist = run_hohmann_cr3bp(alt_leo)
                st.plotly_chart(
                    plot_free_return_compare(_fr2_hist, _hoh_ref_hist, lp),
                    use_container_width=True,
                )
                st.caption(
                    "**青色** = Free-Return 20 天完整軌道（含去程弧、近月點、返程弧）；"
                    "**橙色** = Hohmann 20 天（繼續漂移，無自然返回）。"
                    "  近月點位於月球附近的 L1–L2 連線（x 軸）上；"
                    "  返程弧經過 L4/L5 下方後約第 16.5 天返回地球。"
                    "  ⚠️ 二維共面 CR3BP 教學模型，實際 Artemis II 為 28° 傾角三維軌道。"
                )
                if st.button("💾 儲存 Free-Return 解", use_container_width=True,
                             key="save_fr_exp"):
                    st.session_state["mission_log"].append({
                        "label": f"Free-Return LEO={alt_leo}km 近月={_fr2_peri_act:.0f}km",
                        "dv_total_ms": (_fr2_v - _fr2_v_hoh) * 1000
                                       + plan.dv_tli_kms * 1000,
                        "details": (
                            f"v_TLI={_fr2_v:.4f} km/s, "
                            f"近月={_fr2_peri_act:.0f} km, "
                            f"Jacobi C={_fr2_c:.5f}"
                        ),
                    })
                    st.success("已儲存至任務記錄！")
            else:
                st.info("點擊左側「執行二分法搜尋」按鈕開始計算。首次約 20–40 秒，結果快取後瞬間。")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERT MODE — Interplanetary view
# ═════════════════════════════════════════════════════════════════════════════
else:
    st.header("行星際任務 — Earth → Mars 發射窗口分析")

    SS = st.session_state
    need_run = run_btn or "ip_grid" not in SS

    if need_run:
        with st.status(f"掃描 {dep_n}×{tof_n} 格點…", expanded=True) as _s:
            st.write("執行 Lambert 求解器陣列…")
            grid = run_porkchop(dep_start, dep_end, tof_min, tof_max, dep_n, tof_n)
            best_str = (f"最佳 ΔV = {grid.best_leg.total_dv_kms * 1000:.0f} m/s"
                        if grid.best_leg else "未找到有效解")
            _s.update(label=f"掃描完成 ✓  {best_str}", state="complete",
                      expanded=False)
        SS["ip_grid"] = grid
    else:
        grid = SS["ip_grid"]

    col_map, col_leg = st.columns([3, 1])

    with col_map:
        sel_dep = st.select_slider(
            "選取出發日 (J2000 天)",
            options=[float(f"{d:.1f}") for d in grid.dep_days],
            value=float(f"{grid.best_leg.dep_t_s / 86400:.1f}")
                  if grid.best_leg else float(f"{grid.dep_days[len(grid.dep_days)//2]:.1f}"),
            key="sel_dep",
        )
        sel_tof = st.select_slider(
            "選取飛行時間 (天)",
            options=[float(f"{t:.1f}") for t in grid.tof_days],
            value=float(f"{grid.best_leg.tof_days:.1f}")
                  if grid.best_leg else float(f"{grid.tof_days[len(grid.tof_days)//2]:.1f}"),
            key="sel_tof",
        )
        st.plotly_chart(
            plot_porkchop(grid, sel_dep, sel_tof, show_contours=show_contours),
            use_container_width=True,
        )

    with col_leg:
        st.subheader("選取軌道解算")
        with st.spinner("計算軌道…"):
            leg = run_single_leg(sel_dep, sel_tof)

        if leg:
            v_inf_arr   = float(np.linalg.norm(leg.v_inf_arr))
            r_park      = sim.R_EARTH + 200.0
            r_cap       = R_MARS  + 400.0
            v_park      = math.sqrt(sim.MU_EARTH / r_park)
            two_mu_e_rp = 2.0 * sim.MU_EARTH / r_park
            two_mu_m_rc = 2.0 * MU_MARS / r_cap
            v_cap_circ  = math.sqrt(MU_MARS / r_cap)

            st.metric("C₃",         f"{leg.c3_kms2:.2f} km²/s²")
            st.metric("v∞ 出發",    f"{math.sqrt(max(0, leg.c3_kms2)) * 1000:.0f} m/s")
            st.metric("v∞ 抵達",    f"{v_inf_arr * 1000:.0f} m/s")
            st.divider()
            st.metric("ΔV 出發段",  f"{leg.dv_dep_kms * 1000:.0f} m/s")
            st.metric("ΔV 捕獲段",  f"{leg.dv_arr_kms * 1000:.0f} m/s")
            st.metric("合計 ΔV",    f"{leg.total_dv_kms * 1000:.0f} m/s",
                      delta=f"{'✓ 可行' if leg.total_dv_kms < 4.0 else '偏高'}",
                      delta_color="normal" if leg.total_dv_kms < 4.0 else "inverse")
            st.divider()
            st.metric("TOF", f"{leg.tof_days:.1f} 天")

            _r1e = sim.SimpleEphem.earth_helio(leg.dep_t_s)[0]
            _r1m = sim.SimpleEphem.mars_helio(leg.dep_t_s + leg.tof_days * 86400)[0]
            _cos = float(np.dot(_r1e, _r1m)) / (
                float(np.linalg.norm(_r1e)) * float(np.linalg.norm(_r1m)))
            _dnu = math.degrees(math.acos(min(1.0, max(-1.0, _cos))))
            st.caption(f"出發 J2000+{leg.dep_t_s / 86400:.0f}d\n\n"
                       f"Lambert dnu ≈ {_dnu:.1f}°")

            st.latex(r"\Delta V_{dep} = \sqrt{C_3 + 2\mu_E/r_{park}} - v_{park}")
            st.latex(
                rf"= \sqrt{{{leg.c3_kms2:.2f} + {two_mu_e_rp:.1f}}} - {v_park:.3f}"
                rf"= \mathbf{{{leg.dv_dep_kms * 1000:.0f}}} \text{{ m/s}}"
            )
            st.latex(
                r"\Delta V_{arr} = \sqrt{v_{\infty}^2 + 2\mu_M/r_{cap}} - v_{cap}"
            )
            st.latex(
                rf"= \sqrt{{{v_inf_arr ** 2:.2f} + {two_mu_m_rc:.1f}}} - {v_cap_circ:.3f}"
                rf"= \mathbf{{{leg.dv_arr_kms * 1000:.0f}}} \text{{ m/s}}"
            )
            st.divider()
            if st.button("💾 儲存當前解 (行星際)", use_container_width=True):
                st.session_state["mission_log"].append({
                    "label": (f"Earth→Mars dep=J2000+{sel_dep:.0f}d "
                              f"TOF={sel_tof:.0f}d"),
                    "dv_total_ms": leg.total_dv_kms * 1000,
                    "details": (
                        f"C3={leg.c3_kms2:.2f} km²/s², "
                        f"ΔV_dep={leg.dv_dep_kms * 1000:.0f} m/s, "
                        f"ΔV_arr={leg.dv_arr_kms * 1000:.0f} m/s, "
                        f"dnu={_dnu:.1f}°"
                    ),
                })
                st.success("已儲存至任務記錄！")
        else:
            st.warning("此 (出發日, TOF) 組合 Lambert 求解失敗，\n請調整參數。")

    # Hyperbolic arrival inset
    if leg:
        v_inf_arr = float(np.linalg.norm(leg.v_inf_arr))
        r_cap_km  = R_MARS + 400.0
        st.subheader("火星雙曲線捕獲插圖")
        col_hyp, col_best = st.columns([2, 1])
        with col_hyp:
            st.plotly_chart(
                plot_hyperbolic_arrival(v_inf_arr, r_cap_km),
                use_container_width=True,
            )
        with col_best:
            st.subheader("全局最佳 vs 選取點")
            if grid.best_leg:
                bl = grid.best_leg
                st.metric("全局最佳 ΔV",    f"{bl.total_dv_kms * 1000:.0f} m/s")
                st.metric("全局最佳 出發日",  f"J2000+{bl.dep_t_s / 86400:.0f}d")
                st.metric("全局最佳 TOF",    f"{bl.tof_days:.1f} 天")
                st.metric("全局最佳 C₃",     f"{bl.c3_kms2:.2f} km²/s²")
                st.divider()
                st.metric("選取 ΔV",    f"{leg.total_dv_kms * 1000:.0f} m/s")
                dv_diff = (leg.total_dv_kms - bl.total_dv_kms) * 1000
                st.metric("相差 (選取 − 最佳)",
                          f"{dv_diff:+.0f} m/s",
                          delta_color="inverse" if dv_diff > 50 else "off")

    if grid.best_leg:
        st.divider()
        bl = grid.best_leg
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("最佳出發日",       f"J2000+{bl.dep_t_s / 86400:.0f}d")
        b2.metric("最佳 TOF",         f"{bl.tof_days:.1f} 天")
        b3.metric("最佳 C₃",          f"{bl.c3_kms2:.2f} km²/s²")
        b4.metric("最佳 ΔV 合計",     f"{bl.total_dv_kms * 1000:.0f} m/s")
        b5.metric("ΔV 出發 / 捕獲",
                  f"{bl.dv_dep_kms * 1000:.0f} / {bl.dv_arr_kms * 1000:.0f} m/s")

"""
GMAT-Style Orbital Interception Simulator  v2 — Competition Edition
====================================================================
Modules
-------
  1. Dynamics     – two-body universal variable + RK4 with J2 & exponential drag
  2. Lambert      – two-impulse targeting (universal variable Newton solver)
  3. CW guidance  – Clohessy-Wiltshire rendezvous / intercept burns
  4. Strategy     – single-CW, Lambert 2-impulse, phasing+CW planner
  5. Scoring      – configurable weighted score (time, ΔV, fuel, approach speed)
  6. Scenario I/O – JSON-based mission config + built-in example
  7. Visualization– 3-D trajectories, range, relative speed, fuel, burn schedule

Usage
-----
  python gmat_intercept_sim_v2.py                        # built-in example
  python gmat_intercept_sim_v2.py scenario.json          # custom scenario
  python gmat_intercept_sim_v2.py scenario.json --perturbed  # J2 + drag
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MU_EARTH = 398600.4418   # km³ / s²
R_EARTH  = 6378.137      # km
J2       = 1.08263e-3    # Earth oblateness (dimensionless)
OMEGA_E  = 7.2921150e-5  # rad / s – Earth rotation rate
G0       = 9.80665       # m / s²

# US Standard Atmosphere 1976 – multi-layer exponential
# (base_altitude_km, base_density_kg_m³, scale_height_km)
_ATM_LAYERS: List[Tuple[float, float, float]] = [
    (0,     1.225,      8.44),
    (25,    3.899e-2,   6.49),
    (50,    1.057e-3,   7.47),
    (80,    1.905e-5,   9.50),
    (100,   5.604e-7,   8.70),
    (120,   2.222e-8,   5.78),
    (150,   2.076e-9,   8.00),
    (200,   2.541e-10, 21.44),
    (300,   1.916e-11, 37.11),
    (400,   2.803e-12, 48.72),
    (500,   5.215e-13, 58.13),
    (700,   3.070e-14, 86.75),
    (1000,  3.561e-15, 130.0),
]


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit(v: np.ndarray) -> np.ndarray:
    n = _norm(v)
    return v / n if n > 1e-30 else v


def _stumpff_C(z: float) -> float:
    z = max(z, -200.0)   # clamp to avoid overflow on hyperbolic branch
    if z > 1e-8:
        s = math.sqrt(z);   return (1.0 - math.cos(s)) / z
    if z < -1e-8:
        s = math.sqrt(-z);  return (math.cosh(s) - 1.0) / (-z)
    return 0.5


def _stumpff_S(z: float) -> float:
    z = max(z, -200.0)   # clamp to avoid overflow on hyperbolic branch
    if z > 1e-8:
        s = math.sqrt(z);   return (s - math.sin(s)) / s ** 3
    if z < -1e-8:
        s = math.sqrt(-z);  return (math.sinh(s) - s) / s ** 3
    return 1.0 / 6.0


def _atm_density(h_km: float) -> float:
    """Return atmospheric density in kg / km³ (multi-layer exponential model)."""
    layers = _ATM_LAYERS
    if h_km <= layers[0][0]:
        return layers[0][1] * 1e-9
    for i in range(len(layers) - 1):
        h0, rho0, H = layers[i]
        h1 = layers[i + 1][0]
        if h_km <= h1:
            return rho0 * 1e-9 * math.exp(-(h_km - h0) / H)
    h0, rho0, H = layers[-1]
    return rho0 * 1e-9 * math.exp(-(h_km - h0) / H)


# ─────────────────────────────────────────────────────────────────────────────
# Orbital elements ↔ state vector
# ─────────────────────────────────────────────────────────────────────────────
def coe_to_rv(
    a: float, e: float,
    inc_deg: float, raan_deg: float, argp_deg: float, ta_deg: float,
    mu: float = MU_EARTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Classical orbital elements → ECI (r, v) in km and km/s."""
    inc, raan, argp, ta = (math.radians(x) for x in (inc_deg, raan_deg, argp_deg, ta_deg))
    p = a * (1.0 - e ** 2)
    c_ta, s_ta = math.cos(ta), math.sin(ta)
    denom = 1.0 + e * c_ta
    r_pf = p / denom * np.array([c_ta, s_ta, 0.0])
    v_pf = math.sqrt(mu / p) * np.array([-s_ta, e + c_ta, 0.0])
    cO, sO = math.cos(raan), math.sin(raan)
    ci, si = math.cos(inc),  math.sin(inc)
    cw, sw = math.cos(argp), math.sin(argp)
    Q = np.array([
        [ cO*cw - sO*sw*ci, -cO*sw - sO*cw*ci,  sO*si],
        [ sO*cw + cO*sw*ci, -sO*sw + cO*cw*ci, -cO*si],
        [ sw*si,             cw*si,              ci   ],
    ])
    return Q @ r_pf, Q @ v_pf


def rv_to_coe(
    r_vec: np.ndarray, v_vec: np.ndarray, mu: float = MU_EARTH
) -> Tuple[float, float, float, float, float, float]:
    """ECI (r, v) → (a, e, inc°, raan°, argp°, ta°)."""
    r, v = _norm(r_vec), _norm(v_vec)
    h_vec = np.cross(r_vec, v_vec); h = _norm(h_vec)
    e_vec = ((v**2 - mu/r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e     = _norm(e_vec)
    E     = v**2 / 2.0 - mu / r
    a     = -mu / (2.0 * E) if abs(E) > 1e-10 else float("inf")
    inc   = math.degrees(math.acos(np.clip(h_vec[2] / h, -1, 1)))
    n_vec = np.array([-h_vec[1], h_vec[0], 0.0]); n = _norm(n_vec)
    raan  = 0.0
    if n > 1e-10:
        raan = math.degrees(math.acos(np.clip(n_vec[0] / n, -1, 1)))
        if n_vec[1] < 0: raan = 360.0 - raan
    argp = 0.0
    if n > 1e-10 and e > 1e-10:
        argp = math.degrees(math.acos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1, 1)))
        if e_vec[2] < 0: argp = 360.0 - argp
    ta = 0.0
    if e > 1e-10:
        ta = math.degrees(math.acos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1, 1)))
        if np.dot(r_vec, v_vec) < 0: ta = 360.0 - ta
    return a, e, inc, raan, argp, ta


# ─────────────────────────────────────────────────────────────────────────────
# Frame transformations
# ─────────────────────────────────────────────────────────────────────────────
def _rsw_basis(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return 3×3 matrix [R̂ | Ŝ | Ŵ] as columns (RSW = Radial-Along-track-Cross)."""
    R = _unit(r)
    W = _unit(np.cross(r, v))
    S = np.cross(W, R)
    return np.column_stack((R, S, W))


def _burn_to_eci(frame: str, r: np.ndarray, v: np.ndarray, dv: np.ndarray) -> np.ndarray:
    frame = frame.upper()
    if frame in ("ECI", "MJ2000EQ", "INERTIAL"):
        return dv
    if frame in ("RSW", "LVLH", "RTN", "VNB"):
        return _rsw_basis(r, v) @ dv
    raise ValueError(f"Unsupported burn frame: '{frame}'")


# ─────────────────────────────────────────────────────────────────────────────
# Module 1 – Propagators
# ─────────────────────────────────────────────────────────────────────────────
def _accel_perturbed(r_vec: np.ndarray, v_vec: np.ndarray, cd_am_km2_kg: float) -> np.ndarray:
    """Acceleration (km/s²) from two-body + J2 + drag (optional)."""
    r = _norm(r_vec)
    # Two-body
    a = -MU_EARTH / r**3 * r_vec
    # J2 oblateness
    z = r_vec[2]
    fac = 1.5 * J2 * MU_EARTH * R_EARTH**2 / r**5
    a += fac * np.array([
        r_vec[0] * (5.0*z**2/r**2 - 1.0),
        r_vec[1] * (5.0*z**2/r**2 - 1.0),
        r_vec[2] * (5.0*z**2/r**2 - 3.0),
    ])
    # Atmospheric drag
    if cd_am_km2_kg > 0.0:
        h_km = r - R_EARTH
        rho  = _atm_density(max(h_km, 0.0))
        omega_vec = np.array([0.0, 0.0, OMEGA_E])
        v_rel = v_vec - np.cross(omega_vec, r_vec)
        vr    = _norm(v_rel)
        if vr > 1e-10:
            a -= 0.5 * cd_am_km2_kg * rho * vr * v_rel   # km/s²
    return a


def propagate_rk4(
    r0: np.ndarray, v0: np.ndarray, dt: float,
    cd_am: float = 0.0, substep: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate with 4th-order Runge-Kutta (J2 + exponential drag).

    Parameters
    ----------
    dt      : total propagation interval (s)
    substep : maximum RK4 sub-step size (s).  The inner loop takes
              h = min(substep, remaining), so substep is automatically
              capped to dt — passing step_s < substep is safe and simply
              results in a single RK4 step of size dt.
    cd_am   : Cd × (A/m) in km²/kg; 0 disables drag.
    """
    r, v = r0.copy(), v0.copy()
    remaining = dt
    while remaining > 1e-9:
        h = min(substep, remaining)
        k1r, k1v = v, _accel_perturbed(r, v, cd_am)
        k2r, k2v = v + 0.5*h*k1v, _accel_perturbed(r + 0.5*h*k1r, v + 0.5*h*k1v, cd_am)
        k3r, k3v = v + 0.5*h*k2v, _accel_perturbed(r + 0.5*h*k2r, v + 0.5*h*k2v, cd_am)
        k4r, k4v = v + h*k3v,     _accel_perturbed(r + h*k3r,     v + h*k3v,     cd_am)
        r = r + h/6.0*(k1r + 2*k2r + 2*k3r + k4r)
        v = v + h/6.0*(k1v + 2*k2v + 2*k3v + k4v)
        remaining -= h
    return r, v


def propagate_kepler(
    r0: np.ndarray, v0: np.ndarray, dt: float, mu: float = MU_EARTH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-body Kepler propagation via universal variable (Bate-Mueller-White).

    Convergence note
    ----------------
    The Newton–Raphson loop solves for the universal variable χ (chi), not
    directly for the TOF residual.  The tolerance |Δχ| < 1e-10 is tight enough
    for LEO (χ ~ O(√(km³/s²)) ≈ O(1)) and is consistent with textbook practice.

    Fallback behaviour
    ------------------
    If the loop exits without converging (rare for bound orbits), the last
    Newton iterate is used silently.  For well-behaved LEO/MEO initial conditions
    this is inconsequential; callers that need guaranteed accuracy should use
    propagate_rk4 with J2 disabled (cd_am=0).
    """
    r0n = _norm(r0); v0n = _norm(v0)
    vr0   = float(np.dot(r0, v0) / r0n)
    alpha = 2.0 / r0n - v0n**2 / mu
    # Initial guess for universal variable χ
    if abs(alpha) > 1e-10:
        x = math.sqrt(mu) * abs(alpha) * dt
    else:
        p = _norm(np.cross(r0, v0))**2 / mu
        s = 0.5 * (math.pi/2 - math.atan(3*math.sqrt(mu/p**3) * dt))
        w = math.atan(math.tan(s) ** (1.0/3.0))
        x = math.sqrt(p) * 2.0 / math.tan(2*w)
    for _ in range(100):
        z  = alpha * x**2
        C  = _stumpff_C(z); S = _stumpff_S(z)
        F  = (r0n*vr0/math.sqrt(mu))*x**2*C + (1 - alpha*r0n)*x**3*S + r0n*x - math.sqrt(mu)*dt
        dF = (r0n*vr0/math.sqrt(mu))*x*(1 - z*S) + (1 - alpha*r0n)*x**2*C + r0n
        dx = F / dF; x -= dx
        if abs(dx) < 1e-10:
            break
    z = alpha*x**2; C = _stumpff_C(z); S = _stumpff_S(z)
    f    = 1.0 - x**2 / r0n * C
    g    = dt - x**3 / math.sqrt(mu) * S
    r    = f*r0 + g*v0; rn = _norm(r)
    fdot = math.sqrt(mu) / (rn*r0n) * (alpha*x**3*S - x)
    gdot = 1.0 - x**2 / rn * C
    return r, fdot*r0 + gdot*v0


# ─────────────────────────────────────────────────────────────────────────────
# Module 2 – Lambert solver
# ─────────────────────────────────────────────────────────────────────────────
def _lambert_F(
    z: float, r1: float, r2: float, A: float, mu: float, tof: float
) -> Optional[float]:
    C  = _stumpff_C(z); S = _stumpff_S(z)
    sqC = math.sqrt(max(C, 1e-30))
    y  = r1 + r2 + A*(z*S - 1.0)/sqC
    if y < 1e-6:
        return None
    chi = math.sqrt(y / C) if C > 1e-15 else math.sqrt(y)
    return chi**3 * S + A*math.sqrt(y) - math.sqrt(mu)*tof


def lambert(
    r1_vec: np.ndarray, r2_vec: np.ndarray, tof: float,
    mu: float = MU_EARTH, prograde: bool = True, maxiter: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve Lambert's problem via universal variable (Bate-Mueller-White).

    Scope / limitations
    -------------------
    Single-revolution solver only.  For TOFs longer than one orbital period
    the unique short-arc solution may not exist or may have ΔV > km/s; the
    solver will converge to the lowest-energy single-rev arc regardless.
    Multi-revolution solutions (Izzo / Gooding) are not implemented.

    This solver targets LEO rendezvous: near-circular orbits, moderate chord
    lengths (< ~90° transfer angle), short TOF (minutes to hours).  It is NOT
    suitable for interplanetary transfers or high-eccentricity orbits.

    Derivative method
    -----------------
    dF/dz is computed by finite difference rather than analytically (Vallado
    eq. 7-25).  This is ~2× slower per Newton step but avoids the algebraic
    complexity of the closed-form expression — acceptable for the O(100)
    candidate count in this tool.

    Parameters
    ----------
    r1_vec, r2_vec : departure and arrival position vectors (km)
    tof            : time of flight (s); must be > 0
    prograde       : True for short-way prograde transfer (inclination < 90°)

    Returns
    -------
    v1 : departure velocity at r1_vec (km/s)
    v2 : arrival velocity  at r2_vec (km/s)

    Raises
    ------
    ValueError : degenerate geometry (Δν ≈ 0 or 2π) or singular g coefficient
    """
    r1, r2 = _norm(r1_vec), _norm(r2_vec)
    cos_nu  = float(np.clip(np.dot(r1_vec, r2_vec) / (r1*r2), -1.0, 1.0))
    cross_z = float(np.cross(r1_vec, r2_vec)[2])
    nu      = math.acos(cos_nu)
    if prograde  and cross_z < 0:  nu = 2*math.pi - nu
    if not prograde and cross_z >= 0: nu = 2*math.pi - nu

    A = math.sin(nu) * math.sqrt(r1*r2 / max(1.0 - cos_nu, 1e-12))
    if abs(A) < 1e-8:
        raise ValueError("Lambert: degenerate geometry (Δν ≈ 0 or 2π)")

    # Newton–Raphson with finite-difference derivative (robust but slightly slower)
    z = 0.0
    for _ in range(maxiter):
        z = max(z, -100.0)   # prevent hyperbolic overflow
        F0 = _lambert_F(z, r1, r2, A, mu, tof)
        if F0 is None:
            z += 0.5; continue
        eps = max(abs(z)*1e-6, 1e-7)
        F1 = _lambert_F(z + eps, r1, r2, A, mu, tof)
        if F1 is None:
            z += 0.5; continue
        dFdz = (F1 - F0) / eps
        if abs(dFdz) < 1e-30:
            break
        dz = F0 / dFdz
        dz = max(min(dz, 5.0), -5.0)   # limit step size to prevent oscillation
        z -= dz
        if abs(dz) < 1e-9:
            break

    C  = _stumpff_C(z); S = _stumpff_S(z)
    y  = r1 + r2 + A*(z*S - 1.0)/math.sqrt(max(_stumpff_C(z), 1e-30))
    g  = A * math.sqrt(y / mu)
    if abs(g) < 1e-10:
        raise ValueError("Lambert: singular g coefficient")
    f    = 1.0 - y / r1
    gdot = 1.0 - y / r2
    v1   = (r2_vec - f*r1_vec) / g
    v2   = (gdot*r2_vec - r1_vec) / g
    return v1, v2


# ─────────────────────────────────────────────────────────────────────────────
# Spacecraft model
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Spacecraft:
    """GMAT-like spacecraft with state, propulsion, and optional drag model."""
    name:      str
    r:         np.ndarray    # ECI position (km)
    v:         np.ndarray    # ECI velocity (km/s)
    mass0:     float = 500.0  # initial total mass (kg)
    fuel_mass: float = 120.0  # current propellant mass (kg)
    isp_s:     float = 300.0  # specific impulse (s)
    cd:        float = 2.2    # drag coefficient
    area_m2:   float = 4.0    # drag reference area (m²)
    t:         float = 0.0    # mission elapsed time (s)
    dry_mass:  float = field(init=False)

    def __post_init__(self):
        self.dry_mass = self.mass0 - self.fuel_mass
        if self.dry_mass <= 0:
            raise ValueError(f"{self.name}: dry mass must be positive")

    @property
    def mass(self) -> float:
        return self.dry_mass + self.fuel_mass

    @property
    def _cd_am(self) -> float:
        """Cd × (A/m) in km²/kg (used by drag model)."""
        return self.cd * self.area_m2 * 1e-6 / max(self.mass, 1e-3)

    def propagate(
        self, dt: float,
        perturbed: bool = False,
        substep: float = 30.0,
    ) -> None:
        if perturbed:
            self.r, self.v = propagate_rk4(self.r, self.v, dt, self._cd_am, substep)
        else:
            self.r, self.v = propagate_kepler(self.r, self.v, dt)
        self.t += dt

    def impulsive_burn(self, dv_vec_kms: np.ndarray, frame: str = "RSW") -> float:
        """Apply ImpulsiveBurn; returns fuel consumed (kg). Raises if fuel insufficient."""
        dv_eci = _burn_to_eci(frame, self.r, self.v, dv_vec_kms)
        dv_ms  = _norm(dv_eci) * 1000.0
        if dv_ms < 1e-9:
            return 0.0
        mf   = self.mass / math.exp(dv_ms / (self.isp_s * G0))
        used = self.mass - mf
        if used > self.fuel_mass + 1e-6:
            raise RuntimeError(
                f"{self.name}: insufficient fuel — need {used:.2f} kg, have {self.fuel_mass:.2f} kg"
            )
        self.v        += dv_eci
        self.fuel_mass -= used
        return used

    def copy(self) -> "Spacecraft":
        sc = Spacecraft(
            self.name, self.r.copy(), self.v.copy(),
            self.mass0, self.fuel_mass, self.isp_s, self.cd, self.area_m2,
        )
        sc.dry_mass = self.dry_mass; sc.t = self.t
        return sc


# ─────────────────────────────────────────────────────────────────────────────
# Mission sequence events (GMAT-style)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BurnEvent:
    time_s: float
    dv:     np.ndarray   # ΔV vector in specified frame (km/s)
    frame:  str = "RSW"
    note:   str = ""


@dataclass
class MissionResult:
    success:          bool
    intercept_time_s: Optional[float]
    min_range_km:     float
    final_rel_speed:  float          # km/s at closest approach
    total_dv_kms:     float
    fuel_used_kg:     float
    score:            float
    history:          List[Dict]


# ─────────────────────────────────────────────────────────────────────────────
# Module 5 – Scoring
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ScoringConfig:
    """Configurable scoring weights. All penalties are positive (subtracted)."""
    base_success:        float = 1000.0
    base_fail_coeff:     float = 300.0   # max partial credit on miss
    time_bonus_coeff:    float = 200.0   # bonus for early intercept
    time_bonus_per_600s: float = 1.0     # points per 10-min early
    dv_penalty:          float = 80.0    # per km/s total ΔV
    fuel_penalty:        float = 2.0     # per kg consumed
    approach_penalty:    float = 15.0    # per km/s closing speed at intercept (~0.015 per m/s)

    def compute(
        self,
        success: bool,
        intercept_time_s: Optional[float],
        min_range_km: float,
        total_dv_kms: float,
        fuel_used_kg: float,
        rel_speed_kms: float = 0.0,
    ) -> float:
        base = self.base_success if success else max(0.0, self.base_fail_coeff - min_range_km)
        tb   = (
            max(0.0, self.time_bonus_coeff - intercept_time_s / 600.0 * self.time_bonus_per_600s)
            if intercept_time_s is not None else 0.0
        )
        pen  = (
            self.dv_penalty * total_dv_kms
            + self.fuel_penalty * fuel_used_kg
            + self.approach_penalty * rel_speed_kms
        )
        return base + tb - pen


# ─────────────────────────────────────────────────────────────────────────────
# Mission runner
# ─────────────────────────────────────────────────────────────────────────────
class InterceptMission:
    def __init__(
        self,
        chaser: Spacecraft,
        target: Spacecraft,
        capture_radius_km: float = 10.0,
        time_limit_s:      float = 48 * 3600,
        scoring:           ScoringConfig = None,
        perturbed:         bool = False,
        substep_s:         float = 30.0,
        kill_radius_km:    float = 0.5,
    ):
        self.chaser            = chaser
        self.target            = target
        self.capture_radius_km = capture_radius_km
        self.time_limit_s      = time_limit_s
        self.scoring           = scoring or ScoringConfig()
        self.perturbed         = perturbed
        self.substep_s         = substep_s
        self.kill_radius_km    = kill_radius_km

    def run(self, burns: List[BurnEvent], step_s: float = 60.0) -> MissionResult:
        ch = self.chaser.copy(); tg = self.target.copy()
        burns_sorted = sorted(burns, key=lambda b: b.time_s)
        bi            = 0
        t             = 0.0
        min_range     = float("inf")
        intercept_t   = None
        hist: List[Dict] = []
        total_dv      = 0.0
        m_start       = ch.mass
        final_rv      = 0.0
        closest_rv    = 0.0
        kill_done     = False

        while t <= self.time_limit_s + 1e-9:
            # Apply burns scheduled at or before current time
            while bi < len(burns_sorted) and burns_sorted[bi].time_s <= t + 1e-9:
                b = burns_sorted[bi]
                ch.impulsive_burn(b.dv, b.frame)
                total_dv += _norm(b.dv)
                hist.append({
                    "t_s": t, "event": "burn",
                    "dv_kms": _norm(b.dv),
                    "fuel_used_kg": m_start - ch.mass,
                    "note": b.note,
                })
                bi += 1

            rel_r  = tg.r - ch.r
            rel_v  = tg.v - ch.v
            rng    = _norm(rel_r)
            rv     = _norm(rel_v)

            # Velocity-match kill burn: null relative velocity when close enough
            if not kill_done and rng <= self.kill_radius_km:
                rel_v_mag = _norm(rel_v)
                if rel_v_mag > 0.001:  # > 1 m/s — worth burning
                    dv_kill = -rel_v   # in ECI; burns to match target velocity
                    try:
                        ch.impulsive_burn(dv_kill, "ECI")
                        total_dv += rel_v_mag
                        hist.append({
                            "t_s": t, "event": "burn",
                            "dv_kms": rel_v_mag,
                            "fuel_used_kg": m_start - ch.mass,
                            "note": f"velocity-match kill burn {rel_v_mag*1000:.1f} m/s",
                        })
                        rel_v = tg.v - ch.v   # update after burn
                        rv    = _norm(rel_v)
                    except RuntimeError:
                        pass              # out of fuel — accept residual closing speed
                kill_done = True

            if rng < min_range:
                min_range  = rng
                closest_rv = rv

            hist.append({
                "t_s": t, "event": "state",
                "range_km": rng, "rel_speed_kms": rv,
                "fuel_left_kg": ch.fuel_mass,
                "ch_r": ch.r.tolist(), "tg_r": tg.r.tolist(),
            })

            if rng <= self.capture_radius_km:
                intercept_t = t; final_rv = rv; break

            dt = min(step_s, self.time_limit_s - t)
            if dt <= 0:
                break
            ch.propagate(dt, self.perturbed, self.substep_s)
            tg.propagate(dt, self.perturbed, self.substep_s)
            t += dt

        fuel_used = m_start - ch.mass
        success   = intercept_t is not None
        score     = self.scoring.compute(success, intercept_t, min_range, total_dv, fuel_used, closest_rv)
        return MissionResult(success, intercept_t, min_range, closest_rv, total_dv, fuel_used, score, hist)


# ─────────────────────────────────────────────────────────────────────────────
# Module 3 – CW / Hill relative-motion guidance
# ─────────────────────────────────────────────────────────────────────────────
def cw_stm(n: float, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Clohessy-Wiltshire state transition matrices.
    Returns (Φ_rr, Φ_rv, Φ_vr, Φ_vv) – each 3×3.
    """
    nt  = n * t
    snt = math.sin(nt); cnt = math.cos(nt)
    Prr = np.array([[4-3*cnt, 0, 0], [6*(snt-nt), 1, 0], [0, 0, cnt]])
    Prv = np.array([
        [snt/n,        2*(1-cnt)/n,    0       ],
        [2*(cnt-1)/n, (4*snt-3*nt)/n,  0       ],
        [0,            0,              snt/n   ],
    ])
    Pvr = np.array([[3*n*snt, 0, 0], [6*n*(cnt-1), 0, 0], [0, 0, -n*snt]])
    Pvv = np.array([[cnt, 2*snt, 0], [-2*snt, 4*cnt-3, 0], [0, 0, cnt]])
    return Prr, Prv, Pvr, Pvv


def cw_intercept_burn(
    ch: Spacecraft, tg: Spacecraft, tof_s: float
) -> np.ndarray:
    """
    Single CW burn at t=0 to place chaser at target position after tof_s.
    Returns ΔV in RSW frame (km/s).
    """
    n   = math.sqrt(MU_EARTH / _norm(ch.r)**3)
    B   = _rsw_basis(ch.r, ch.v)
    dr0 = B.T @ (tg.r - ch.r)      # relative position in RSW
    dv0 = B.T @ (tg.v - ch.v)      # relative velocity in RSW
    Prr, Prv, _, _ = cw_stm(n, tof_s)
    rhs = -Prr @ dr0
    req = np.linalg.solve(Prv, rhs)
    return req - dv0    # ΔV to apply to chaser (RSW)


def cw_rendezvous_burns(
    ch: Spacecraft, tg: Spacecraft, tof_s: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two CW burns for full rendezvous (position and velocity match after tof_s).
    Returns (dv1_rsw, dv2_rsw) in km/s.
    """
    n   = math.sqrt(MU_EARTH / _norm(ch.r)**3)
    B   = _rsw_basis(ch.r, ch.v)
    dr0 = B.T @ (tg.r - ch.r)
    dv0 = B.T @ (tg.v - ch.v)
    Prr, Prv, Pvr, Pvv = cw_stm(n, tof_s)
    req  = np.linalg.solve(Prv, -Prr @ dr0)
    dv1  = req - dv0
    vf   = Pvr @ dr0 + Pvv @ req
    dv2  = -vf   # arrival kill burn
    return dv1, dv2


def cw_applicable(
    ch: Spacecraft, tg: Spacecraft,
    sep_r_max: float = 0.1,
    vrel_nr_max: float = 2.0,
) -> bool:
    """
    Check whether the CW (Clohessy-Wiltshire) linear model is physically valid.

    The Hill-frame linearisation assumes:
      (a) reference orbit is near-circular
      (b) relative separation r_rel ≪ semi-major axis  (sep / r < sep_r_max)
      (c) relative velocity is not excessively large compared to n × sep
          (prevents gross over-prediction of required ΔV)

    These are order-of-magnitude guards, not hard physical cutoffs.  They
    complement the absolute 500 km ceiling (CW_MAX_SEP_KM in search_plans)
    with a dimensionless check that scales with the orbital radius.

    Parameters
    ----------
    sep_r_max   : max allowed sep/r ratio (default 0.1 → 10 % of orbit radius)
    vrel_nr_max : max allowed v_rel / (n × sep) ratio (default 2.0)
    """
    r   = _norm(ch.r)
    sep = _norm(tg.r - ch.r)
    n   = math.sqrt(MU_EARTH / r**3)
    if sep > 500.0:
        return False
    if sep / r > sep_r_max:
        return False
    if sep > 1e-3:
        v_rel = _norm(tg.v - ch.v)
        if v_rel / (n * sep) > vrel_nr_max:
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Module 4 – Strategy planners
# ─────────────────────────────────────────────────────────────────────────────
def _lambert_two_impulse(
    ch: Spacecraft, tg: Spacecraft, tof_s: float, prograde: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Lambert targeting: returns (dv1_eci, dv2_eci)."""
    tg_fut = tg.copy(); tg_fut.propagate(tof_s)
    v1_new, v2_arr = lambert(ch.r, tg_fut.r, tof_s, prograde=prograde)
    dv1 = v1_new - ch.v
    dv2 = tg_fut.v - v2_arr  # braking to match target velocity
    return dv1, dv2


def find_resonant_phasing(
    chaser: Spacecraft,
    target: Spacecraft,
    wait_s: float = 0.0,
    n_orbits_list: Optional[List[float]] = None,
) -> List[Dict]:
    """
    Analytically solve for phasing orbits that close a phase gap exactly.

    For each candidate orbit count N, computes the unique semi-major axis a_ph
    such that after exactly N revolutions of the phasing orbit the chaser has
    gained exactly *gap* radians on the target.  No scanning over Δh — the
    solution is direct.

    Returns a list of dicts sorted by |ΔV|, each with keys:
        n_orbits, a_ph_km, dv_kms, t_phase_s, dh_km, feasible
    """
    if n_orbits_list is None:
        n_orbits_list = list(np.arange(0.5, 20.1, 0.5))

    ch0 = chaser.copy(); tg0 = target.copy()
    if wait_s > 0:
        ch0.propagate(wait_s); tg0.propagate(wait_s)

    r0  = _norm(ch0.r)
    n_c = math.sqrt(MU_EARTH / r0**3)
    T_c = 2 * math.pi / n_c
    n_t = math.sqrt(MU_EARTH / _norm(tg0.r)**3)
    v_c = _norm(ch0.v)

    h_c = _unit(np.cross(ch0.r, ch0.v))
    cr  = float(np.dot(np.cross(_unit(ch0.r), _unit(tg0.r)), h_c))
    gap = math.acos(np.clip(np.dot(_unit(ch0.r), _unit(tg0.r)), -1, 1))
    if cr < 0:
        gap = -gap

    results = []
    for N in n_orbits_list:
        if N <= 0:
            continue
        # Always anchor on chaser's n_c (not target's n_t).
        # gap > 0: target ahead  → chaser must speed up  → n_ph > n_c → lower a_ph
        # gap < 0: target behind → chaser must slow down → n_ph < n_c → raise a_ph
        # Physics: (n_ph − n_c) × N × T_c = gap  →  delta_n = gap / (N × T_c)
        delta_n = gap / (N * T_c)
        n_ph    = n_c + delta_n

        if n_ph <= 0:
            continue
        a_ph     = (MU_EARTH / n_ph**2) ** (1.0 / 3.0)
        T_ph     = 2 * math.pi / n_ph
        t_phase  = N * T_ph
        dh_km    = r0 - a_ph           # negative means raising orbit

        v_ph_at_r = math.sqrt(max(0.0, MU_EARTH * (2.0 / r0 - 1.0 / a_ph)))
        dv_kms   = abs(v_ph_at_r - v_c)

        feasible = (
            a_ph > R_EARTH + 200
            and dv_kms < 0.5          # PHASE_MAX_DV_KMS
            and t_phase + wait_s < 86400 * 2
        )
        results.append({
            "n_orbits":  N,
            "a_ph_km":   a_ph,
            "dh_km":     dh_km,
            "dv_kms":    dv_kms,
            "t_phase_s": t_phase,
            "gap_rad":   gap,
            "feasible":  feasible,
        })

    results.sort(key=lambda x: x["dv_kms"])
    return results


def search_plans(
    chaser:            Spacecraft,
    target:            Spacecraft,
    time_limit_s:      float,
    fuel_cap_kg:       float,
    capture_radius_km: float = 10.0,
    scoring:           ScoringConfig = None,
    perturbed:         bool = False,
    step_s:            float = 60.0,
    verbose:           bool = True,
) -> Tuple[Optional[MissionResult], List[BurnEvent], List]:
    """
    Search over three maneuver strategies and return the best plan.

    Strategies
    ----------
    A – Single CW intercept burn (vary wait time + TOF)
    B – Lambert two-impulse transfer (prograde + retrograde)
    C – Phasing coast + CW terminal rendezvous

    Returns
    -------
    (best_result, best_burns, all_candidates)
    """
    scoring = scoring or ScoringConfig()
    mission = InterceptMission(
        chaser, target, capture_radius_km, time_limit_s, scoring, perturbed
    )
    candidates: List[Tuple[str, List[BurnEvent], MissionResult]] = []

    # TOF grid covers sub-orbital to multi-day transfers; finer grid for short TOFs
    WAIT_H = [0, 0.5, 1, 2, 3, 4, 6, 8, 12, 18, 24]
    TOF_H  = [0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 18, 24, 30, 36, 42, 48]

    # CW is only valid for close-range relative motion (< 500 km separation)
    CW_MAX_SEP_KM   = 500.0
    # Per-impulse ΔV guards (pre-filter before running the full simulation)
    LAM_MAX_DV_KMS  = 2.0   # km/s per Lambert impulse
    CW_MAX_DV_KMS   = 1.0   # km/s per CW terminal impulse
    PHASE_MAX_DV_KMS = 0.5  # km/s per phasing impulse (Hohmann-style, should be small)

    def _try(burns_list: List[BurnEvent], label: str) -> None:
        try:
            res = mission.run(burns_list, step_s)
        except RuntimeError:
            return
        if res.fuel_used_kg <= fuel_cap_kg:
            candidates.append((label, burns_list, res))
            if verbose:
                sym = "OK" if res.success else "--"
                print(
                    f"  {sym} {label:<46} "
                    f"score={res.score:7.1f}  "
                    f"dv={res.total_dv_kms*1000:6.1f} m/s  "
                    f"fuel={res.fuel_used_kg:5.1f} kg  "
                    f"rng={res.min_range_km:7.1f} km"
                )

    # ── Strategy A: single CW intercept burn (close-range only) ─────────────
    if verbose:
        print("\n── Strategy A: CW intercept  [valid: sep<500km, sep/r<0.1] ──────────")
    for wh in WAIT_H:
        for th in TOF_H:
            t1 = wh * 3600; t2 = th * 3600
            if t1 + t2 > time_limit_s: continue
            ch0 = chaser.copy(); tg0 = target.copy()
            if t1 > 0: ch0.propagate(t1); tg0.propagate(t1)
            if not cw_applicable(ch0, tg0): continue   # CW validity (sep, sep/r, v_rel)
            try:
                dv1 = cw_intercept_burn(ch0, tg0, t2)
            except np.linalg.LinAlgError:
                continue
            if _norm(dv1) > CW_MAX_DV_KMS: continue
            _try(
                [BurnEvent(t1, dv1, "RSW", f"CW-intercept wt={wh}h tof={th}h")],
                f"CW-intercept   wt={wh:4.1f}h  tof={th:4.0f}h",
            )

    # ── Strategy B: Lambert two-impulse (prograde + retrograde) ─────────────
    if verbose:
        print("\n── Strategy B: Lambert two-impulse ──────────────────────────────────")
    for wh in WAIT_H:
        for th in TOF_H:
            t1 = wh * 3600; t2 = th * 3600
            if t1 + t2 > time_limit_s: continue
            if t2 < 300.0: continue   # minimum TOF: 5 min (avoids near-zero chord)
            ch0 = chaser.copy(); tg0 = target.copy()
            if t1 > 0: ch0.propagate(t1); tg0.propagate(t1)
            for prograde in (True, False):
                try:
                    dv1, dv2 = _lambert_two_impulse(ch0, tg0, t2, prograde)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if _norm(dv1) > LAM_MAX_DV_KMS: continue
                if _norm(dv2) > LAM_MAX_DV_KMS: continue
                tag = "pro" if prograde else "retro"
                _try(
                    [
                        BurnEvent(t1,      dv1, "ECI", f"Lambert-1 {tag} wt={wh}h tof={th}h"),
                        BurnEvent(t1 + t2, dv2, "ECI", f"Lambert-2 {tag} arrival"),
                    ],
                    f"Lambert-{tag:<5} wt={wh:4.1f}h  tof={th:4.0f}h",
                )

    # ── Strategy C: Lambert depart + CW terminal rendezvous ──────────────────
    # Coast to a Lambert transfer that brings chaser within CW range, then CW.
    if verbose:
        print("\n── Strategy C: Lambert approach + CW terminal ───────────────────────")
    for wh in WAIT_H:
        for th_lam in [2, 4, 6, 8, 12, 18, 24]:   # Lambert leg
            for th_cw in [0.5, 1, 2, 4]:           # CW terminal leg
                t1 = wh * 3600
                t2 = th_lam * 3600
                t3 = th_cw * 3600
                if t1 + t2 + t3 > time_limit_s: continue
                ch0 = chaser.copy(); tg0 = target.copy()
                if t1 > 0: ch0.propagate(t1); tg0.propagate(t1)
                # Propagate to intermediate Lambert arrival point
                tg_mid = tg0.copy(); tg_mid.propagate(t2)
                try:
                    v1_new, _v2 = lambert(ch0.r, tg_mid.r, t2)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                dv1 = v1_new - ch0.v
                if _norm(dv1) > LAM_MAX_DV_KMS: continue
                # At t1+t2: chaser arrives at tg_mid.r; compute CW terminal burn
                ch_mid = ch0.copy(); ch_mid.v = v1_new; ch_mid.propagate(t2)
                if not cw_applicable(ch_mid, tg_mid): continue
                try:
                    dv_cw, dv_arr = cw_rendezvous_burns(ch_mid, tg_mid, t3)
                except np.linalg.LinAlgError:
                    continue
                if _norm(dv_cw) > CW_MAX_DV_KMS: continue
                burns_list = [BurnEvent(t1, dv1, "ECI", f"Lambert approach wt={wh}h")]
                if _norm(dv_cw) > 1e-6:
                    burns_list.append(BurnEvent(t1+t2, dv_cw, "RSW", f"CW terminal t={th_cw}h"))
                if _norm(dv_arr) > 1e-6:
                    burns_list.append(BurnEvent(t1+t2+t3, dv_arr, "RSW", "CW arrival kill"))
                _try(burns_list, f"Lam+CW  wt={wh:3.1f}h Llg={th_lam:2.0f}h CWlg={th_cw:.1f}h")

    # ── Strategy D: Phasing orbit (Hohmann lower → coast N orbits → raise) ───
    # Best for large phase gaps where direct Lambert requires huge ΔV.
    # Lower the orbit so the chaser gains angular velocity on the target,
    # then raise back once the gap is closed.  Total ΔV is typically < 50 m/s.
    if verbose:
        print("\n── Strategy D: Phasing orbit (lower → coast → raise) ────────────────")
    # Fine grid: fractional orbit counts and computed Δh from altitude drop table
    _phase_waits   = [0.0, 0.5, 1.0, 2.0]
    _phase_n_orbs  = list(np.arange(0.5, 15.1, 0.5))    # 0.5, 1.0, 1.5 … 15.0
    _phase_dh_km   = list(np.arange(5.0, 51.0, 5.0))    # 5, 10, … 50 km altitude drops

    for wh in _phase_waits:
        t1 = wh * 3600
        ch0 = chaser.copy(); tg0 = target.copy()
        if t1 > 0: ch0.propagate(t1); tg0.propagate(t1)

        # Current orbital parameters (computed once per wait-time)
        r0   = _norm(ch0.r)
        n_c  = math.sqrt(MU_EARTH / r0**3)
        T_c  = 2*math.pi / n_c

        # Phase gap: signed angle by which target is ahead (+) or behind (-)
        h_c  = _unit(np.cross(ch0.r, ch0.v))
        cr   = float(np.dot(np.cross(_unit(ch0.r), _unit(tg0.r)), h_c))
        gap  = math.acos(np.clip(np.dot(_unit(ch0.r), _unit(tg0.r)), -1, 1))
        if cr < 0: gap = -gap
        if gap <= 0: continue   # target trailing — lowering won't help

        # ── Sub-loop A: orbit-count driven (analytical resonant solution) ──
        # Same reference as find_resonant_phasing: anchor on chaser n_c.
        # delta_n = gap / (N × T_c); n_ph = n_c + delta_n (signed).
        for n_phase in _phase_n_orbs:
            delta_n   = gap / (n_phase * T_c)
            n_ph      = n_c + delta_n
            a_ph      = (MU_EARTH / n_ph**2)**(1/3)
            T_ph      = 2*math.pi / n_ph
            t_phase_dur = n_phase * T_ph

            if a_ph < R_EARTH + 200: continue
            if a_ph > r0: continue
            if t1 + t_phase_dur > time_limit_s - 300: continue

            v_c  = _norm(ch0.v)
            v_ph_at_r = math.sqrt(MU_EARTH * (2.0/r0 - 1.0/a_ph))
            dv1_s = v_ph_at_r - v_c
            dv1   = np.array([0.0, dv1_s, 0.0])
            if abs(dv1_s) > PHASE_MAX_DV_KMS: continue
            dv2 = np.array([0.0, -dv1_s, 0.0])
            t2  = t1 + t_phase_dur

            tag = f"n={n_phase:.1f}orb wt={wh}h"
            burns_2 = [
                BurnEvent(t1, dv1, "RSW", f"Phase lower {tag}"),
                BurnEvent(t2, dv2, "RSW", f"Phase raise {tag}"),
            ]
            _try(burns_2, f"Phasing     wt={wh:3.1f}h n={n_phase:4.1f}orb dh={abs(r0-a_ph)*1000:.0f}m")

            for t_term_h in [0.25, 0.5, 1.0, 2.0]:
                t3 = t_term_h * 3600
                if t2 + t3 > time_limit_s: continue
                ch_post = ch0.copy()
                ch_post.impulsive_burn(dv1, "RSW")
                ch_post.propagate(t_phase_dur)
                ch_post.impulsive_burn(dv2, "RSW")
                tg_post = tg0.copy(); tg_post.propagate(t_phase_dur)
                if not cw_applicable(ch_post, tg_post): continue
                try:
                    dv_term = cw_intercept_burn(ch_post, tg_post, t3)
                except np.linalg.LinAlgError:
                    continue
                if _norm(dv_term) > CW_MAX_DV_KMS: continue
                burns_3 = [
                    BurnEvent(t1, dv1, "RSW", f"Phase lower {tag}"),
                    BurnEvent(t2, dv2, "RSW", f"Phase raise {tag}"),
                    BurnEvent(t2, dv_term, "RSW", f"Terminal CW t={t_term_h}h"),
                ]
                _try(burns_3, f"Phase+CW    wt={wh:3.1f}h n={n_phase:4.1f}orb trm={t_term_h}h")

        # ── Sub-loop B: altitude-drop driven (scan Δh, derive n_orbits needed) ──
        for dh_km in _phase_dh_km:
            a_ph = r0 - dh_km                            # lower SMA in km
            if a_ph < R_EARTH + 200: continue

            T_ph      = 2*math.pi * math.sqrt(a_ph**3 / MU_EARTH)
            n_ph_mo   = 2*math.pi / T_ph                 # phasing mean motion
            delta_n   = n_ph_mo - n_c                    # extra angular rate gained
            if delta_n <= 0: continue

            n_orbits_needed = gap / (delta_n * T_c)      # non-integer orbit count
            if n_orbits_needed < 0.25 or n_orbits_needed > 20: continue

            # Round to nearest half-orbit for clean burn scheduling
            n_orbits = round(n_orbits_needed * 2) / 2.0
            if n_orbits < 0.5: n_orbits = 0.5
            t_phase_dur = n_orbits * T_ph
            if t1 + t_phase_dur > time_limit_s - 300: continue

            v_c  = _norm(ch0.v)
            v_ph_at_r = math.sqrt(MU_EARTH * (2.0/r0 - 1.0/a_ph))
            dv1_s = v_ph_at_r - v_c
            dv1   = np.array([0.0, dv1_s, 0.0])
            if abs(dv1_s) > PHASE_MAX_DV_KMS: continue
            dv2 = np.array([0.0, -dv1_s, 0.0])
            t2  = t1 + t_phase_dur

            tag = f"dh={dh_km:.0f}km wt={wh}h"
            burns_2 = [
                BurnEvent(t1, dv1, "RSW", f"Phase lower {tag}"),
                BurnEvent(t2, dv2, "RSW", f"Phase raise {tag}"),
            ]
            _try(burns_2, f"PhasingDh   wt={wh:3.1f}h dh={dh_km:4.0f}km n~{n_orbits:.1f}orb")

            for t_term_h in [0.25, 0.5, 1.0, 2.0]:
                t3 = t_term_h * 3600
                if t2 + t3 > time_limit_s: continue
                ch_post = ch0.copy()
                ch_post.impulsive_burn(dv1, "RSW")
                ch_post.propagate(t_phase_dur)
                ch_post.impulsive_burn(dv2, "RSW")
                tg_post = tg0.copy(); tg_post.propagate(t_phase_dur)
                if not cw_applicable(ch_post, tg_post): continue
                try:
                    dv_term = cw_intercept_burn(ch_post, tg_post, t3)
                except np.linalg.LinAlgError:
                    continue
                if _norm(dv_term) > CW_MAX_DV_KMS: continue
                burns_3 = [
                    BurnEvent(t1, dv1, "RSW", f"Phase lower {tag}"),
                    BurnEvent(t2, dv2, "RSW", f"Phase raise {tag}"),
                    BurnEvent(t2, dv_term, "RSW", f"Terminal CW t={t_term_h}h"),
                ]
                _try(burns_3, f"PhasDhCW    wt={wh:3.1f}h dh={dh_km:4.0f}km trm={t_term_h}h")

    if not candidates:
        return None, [], []

    best = max(candidates, key=lambda c: c[2].score)
    return best[2], best[1], candidates


# ─────────────────────────────────────────────────────────────────────────────
# Module 7 – Visualization
# ─────────────────────────────────────────────────────────────────────────────
def visualize(
    result: MissionResult,
    burns:  List[BurnEvent],
    title:  str = "Interception Mission",
    save_path: str = "mission_summary.png",
) -> None:
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("[viz] matplotlib not available — skipping visualization")
        return

    states   = [h for h in result.history if h["event"] == "state"]
    burn_evt = [h for h in result.history if h["event"] == "burn"]

    t_h      = np.array([s["t_s"] / 3600.0 for s in states])
    rng_km   = np.array([s["range_km"]      for s in states])
    rel_v    = np.array([s["rel_speed_kms"] for s in states])
    fuel_kg  = np.array([s["fuel_left_kg"]  for s in states])
    ch_pos   = np.array([s["ch_r"]          for s in states])
    tg_pos   = np.array([s["tg_r"]          for s in states])

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ── 3-D trajectories (left panel, tall) ───────────────────────────────
    ax3d = fig.add_subplot(2, 3, (1, 4), projection="3d")
    ax3d.plot(*ch_pos.T, "b-", lw=1.2, label="Chaser")
    ax3d.plot(*tg_pos.T, "r--", lw=1.0, label="Target")
    ax3d.plot(*ch_pos[0], "b^", ms=8)
    ax3d.plot(*tg_pos[0], "rs", ms=8)
    if result.success:
        idx = int(np.argmin(rng_km))
        ax3d.plot(*ch_pos[idx], "g*", ms=14, zorder=5, label="Intercept")
    for bh in burn_evt:
        tidx = int(np.argmin(abs(t_h - bh["t_s"] / 3600.0)))
        ax3d.plot(*ch_pos[tidx], "ko", ms=5)
    # Earth sphere
    u = np.linspace(0, 2*np.pi, 40); v_ = np.linspace(0, np.pi, 20)
    xe = R_EARTH * np.outer(np.cos(u), np.sin(v_))
    ye = R_EARTH * np.outer(np.sin(u), np.sin(v_))
    ze = R_EARTH * np.outer(np.ones_like(u), np.cos(v_))
    ax3d.plot_surface(xe, ye, ze, alpha=0.12, color="deepskyblue", linewidth=0)
    ax3d.set_xlabel("X (km)"); ax3d.set_ylabel("Y (km)"); ax3d.set_zlabel("Z (km)")
    ax3d.legend(fontsize=7, loc="upper left"); ax3d.set_title("3-D Trajectories")

    # ── Range vs time ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.semilogy(t_h, rng_km, "k-", lw=1.5)
    for bh in burn_evt:
        ax2.axvline(bh["t_s"] / 3600.0, color="orange", ls="--", lw=1)
    ax2.axhline(rng_km.min(), color="green", ls=":", lw=1.2,
                label=f"Min: {rng_km.min():.1f} km")
    ax2.set_xlabel("Time (h)"); ax2.set_ylabel("Range (km)")
    ax2.set_title("Relative Range"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # ── Closing speed vs time ──────────────────────────────────────────────
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t_h, rel_v * 1000.0, "m-", lw=1.5)
    ax3.set_xlabel("Time (h)"); ax3.set_ylabel("Rel. speed (m/s)")
    ax3.set_title("Closing Speed"); ax3.grid(True, alpha=0.3)

    # ── Fuel remaining ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(2, 3, 5)
    ax4.plot(t_h, fuel_kg, "g-", lw=1.5)
    ax4.set_xlabel("Time (h)"); ax4.set_ylabel("Fuel left (kg)")
    ax4.set_title("Propellant Budget"); ax4.grid(True, alpha=0.3)

    # ── Burn schedule ──────────────────────────────────────────────────────
    ax5 = fig.add_subplot(2, 3, 6)
    if burns:
        bt_h  = [b.time_s / 3600.0 for b in burns]
        dv_ms = [_norm(b.dv) * 1000.0 for b in burns]
        ax5.bar(range(len(bt_h)), dv_ms, color="orange", edgecolor="k")
        ax5.set_xticks(range(len(bt_h)))
        ax5.set_xticklabels([f"t={t:.1f}h" for t in bt_h], rotation=30, fontsize=8)
        ax5.set_ylabel("ΔV (m/s)"); ax5.set_title("Burn Schedule")
        ax5.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[viz] Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Module 6 – Scenario I/O
# ─────────────────────────────────────────────────────────────────────────────
_EXAMPLE_SCENARIO: Dict = {
    # Scenario: coplanar nearby orbits, ~35° phase gap
    # Chaser is behind the target in similar LEO.  Both orbits differ by only
    # 0.5° inclination and ~20 km altitude, keeping ΔV within the fuel budget.
    # Challenge: intercept within 24 h using minimum propellant.
    "chaser": {
        "coe": {
            "a": R_EARTH + 500, "e": 0.001,
            "inc_deg": 28.5, "raan_deg": 45.0,
            "argp_deg": 0.0,  "ta_deg":  0.0,
        },
        "mass0_kg": 600, "fuel_mass_kg": 140, "isp_s": 315,
        "cd": 2.2, "area_m2": 6.0,
    },
    "target": {
        "coe": {
            "a": R_EARTH + 520, "e": 0.003,
            "inc_deg": 29.0,  "raan_deg": 45.5,
            "argp_deg": 0.0,  "ta_deg": 35.0,   # ~35° ahead of chaser
        },
        "mass0_kg": 900, "fuel_mass_kg": 0, "isp_s": 1,
    },
    "mission": {
        "time_limit_s":      24 * 3600,
        "capture_radius_km": 15.0,
        "fuel_cap_kg":       85.0,
        "perturbed":         False,
        "step_s":            30.0,
    },
    "scoring": {
        "base_success":    1000.0,
        "dv_penalty":        80.0,
        "fuel_penalty":       2.0,
        "approach_penalty":  15.0,
    },
}


def _build_spacecraft(cfg: Dict, name: str) -> Spacecraft:
    if "coe" in cfg:
        r, v = coe_to_rv(**cfg["coe"])
    else:
        r = np.array(cfg["r_km"]); v = np.array(cfg["v_kms"])
    return Spacecraft(
        name, r, v,
        mass0=cfg.get("mass0_kg", 500),
        fuel_mass=cfg.get("fuel_mass_kg", 120),
        isp_s=cfg.get("isp_s", 300),
        cd=cfg.get("cd", 2.2),
        area_m2=cfg.get("area_m2", 4.0),
    )


def load_scenario(path: str) -> Dict:
    with open(path) as fh:
        return json.load(fh)


def save_example_scenario(path: str = "example_scenario.json") -> None:
    with open(path, "w") as fh:
        json.dump(_EXAMPLE_SCENARIO, fh, indent=2)
    print(f"[io] Example scenario written → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def run(
    scenario:       Dict = None,
    perturbed:      bool = False,
    show_viz:       bool = True,
    save_viz_path:  str  = "mission_summary.png",
) -> Tuple[Optional[MissionResult], List[BurnEvent], List]:
    cfg        = scenario or _EXAMPLE_SCENARIO
    mc         = cfg.get("mission", {})
    chaser     = _build_spacecraft(cfg["chaser"], "Chaser")
    target     = _build_spacecraft(cfg["target"], "Target")
    scoring    = ScoringConfig(**cfg.get("scoring", {}))
    time_limit = mc.get("time_limit_s",      48 * 3600)
    fuel_cap   = mc.get("fuel_cap_kg",        80.0)
    cap_r      = mc.get("capture_radius_km",  15.0)
    perturbed  = perturbed or mc.get("perturbed", False)
    step_s     = mc.get("step_s",             60.0)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║    GMAT-Style Orbital Interception Simulator  v2            ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    a_ch, e_ch, i_ch, Ra_ch, w_ch, ta_ch = rv_to_coe(chaser.r, chaser.v)
    a_tg, e_tg, i_tg, Ra_tg, w_tg, ta_tg = rv_to_coe(target.r, target.v)
    print(f"\nChaser : a={a_ch:.1f} km  e={e_ch:.4f}  i={i_ch:.2f}°  "
          f"RAAN={Ra_ch:.1f}°  ω={w_ch:.1f}°  ν={ta_ch:.1f}°")
    print(f"         mass={chaser.mass:.0f} kg  fuel={chaser.fuel_mass:.0f} kg  Isp={chaser.isp_s:.0f} s")
    print(f"Target : a={a_tg:.1f} km  e={e_tg:.4f}  i={i_tg:.2f}°  "
          f"RAAN={Ra_tg:.1f}°  ω={w_tg:.1f}°  ν={ta_tg:.1f}°")
    print(f"\nTime limit      : {time_limit/3600:.0f} h")
    print(f"Fuel budget     : {fuel_cap:.0f} kg")
    print(f"Capture radius  : {cap_r:.0f} km")
    print(f"Dynamics model  : {'Perturbed (J2 + drag)' if perturbed else 'Two-body Keplerian'}")

    # Show analytical resonant solutions before the full search
    print("\n── Analytical resonant phasing solutions (sorted by ΔV) ─────────────")
    for wh in [0.0, 0.5, 1.0, 2.0]:
        rsols = find_resonant_phasing(chaser, target, wait_s=wh * 3600)
        feasible = [s for s in rsols if s["feasible"]][:3]
        if feasible:
            print(f"  wt={wh:.1f}h: " + "  |  ".join(
                f"N={s['n_orbits']:.1f}orb dh={s['dh_km']*1000:.0f}m dv={s['dv_kms']*1000:.1f}m/s"
                for s in feasible
            ))

    best_res, best_burns, candidates = search_plans(
        chaser, target, time_limit, fuel_cap, cap_r,
        scoring, perturbed, step_s, verbose=True,
    )

    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  Candidates evaluated : {len(candidates)}")

    if best_res is None:
        print("  No feasible plan found within constraints.")
        return None, [], []

    print(f"\n{'─'*20} BEST PLAN {'─'*20}")
    print(f"  Success          : {best_res.success}")
    ti = best_res.intercept_time_s
    if ti is not None:
        h, m, s = int(ti//3600), int((ti%3600)//60), int(ti%60)
        print(f"  Intercept time   : {ti/3600:.3f} h  ({h:02d}:{m:02d}:{s:02d})")
    print(f"  Min range        : {best_res.min_range_km:.3f} km")
    print(f"  Closing speed    : {best_res.final_rel_speed*1000:.2f} m/s")
    print(f"  Total ΔV         : {best_res.total_dv_kms*1000:.2f} m/s")
    print(f"  Fuel consumed    : {best_res.fuel_used_kg:.2f} kg")
    print(f"  Score            : {best_res.score:.2f}")
    print(f"  Burns ({len(best_burns)}):")
    for i, b in enumerate(best_burns, 1):
        print(f"    #{i}  t={b.time_s/3600:.3f} h  frame={b.frame:4s}  "
              f"|ΔV|={_norm(b.dv)*1000:.2f} m/s  ({b.note})")

    if show_viz:
        visualize(
            best_res, best_burns,
            title=f"Best Plan — Score: {best_res.score:.1f}",
            save_path=save_viz_path,
        )

    return best_res, best_burns, candidates


if __name__ == "__main__":
    _perturbed = "--perturbed" in sys.argv
    _scenario_file = next((a for a in sys.argv[1:] if not a.startswith("--")), None)

    if _scenario_file:
        print(f"[io] Loading scenario: {_scenario_file}")
        _cfg = load_scenario(_scenario_file)
    else:
        _cfg = None

    run(scenario=_cfg, perturbed=_perturbed)

#!/usr/bin/env python3
"""
gmat_intercept_sim_v2_CR3BP.py
=================================
Cislunar and interplanetary extension of gmat_intercept_sim_v2.py.

Extends the existing LEO rendezvous engine to:
  • Cislunar space  — Earth-Moon CR3BP propagator (rotating frame),
                      patched-conics TLI / free-return / LOI planning
  • Interplanetary  — heliocentric Lambert (separate scale-correct solver),
                      departure/arrival hyperbola, Earth→Mars porkchop plot

Architecture decisions (based on technical review)
---------------------------------------------------
1. CR3BP is a *separate propagator class*, not a patch to the existing RK4.
   The existing LEO engine is imported and left unchanged.
2. Patched conics uses *sequential segments* (each with one central body);
   no SOI-switching hack inside the integration loop.
3. A new ``lambert_izzo()`` (Izzo 2015) handles interplanetary Lambert
   problems for all transfer angles, including long-way (Δν > π) transfers.
   The v2 ``lambert()`` remains for LEO use.
4. Ephemeris uses simplified circular-orbit approximations (no external deps),
   clearly marked as educational accuracy only.
5. All interplanetary code lives here, not in gmat_intercept_sim_v2.py, so the
   LEO engine stays lean.

Units: km, km/s, seconds — except where CR3BP normalized units are used.

Run standalone demo
-------------------
    python gmat_intercept_sim_v2_CR3BP.py
"""
from __future__ import annotations

import sys, math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Force UTF-8 output so box-drawing, bullets, and arrows print correctly on
# Windows terminals whose default codepage (cp950/cp932/etc.) cannot encode them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

# ── Import LEO engine ─────────────────────────────────────────────────────────
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gmat_intercept_sim_v2 import (
    _norm,
    MU_EARTH, R_EARTH,
    coe_to_rv, rv_to_coe,
)

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────
MU_SUN  = 1.32712440018e11   # km³/s²  (heliocentric)
MU_MOON = 4902.800116        # km³/s²
MU_MARS = 42828.375          # km³/s²

R_SUN   = 695700.0           # km
R_MOON  = 1737.1             # km
R_MARS  = 3389.5             # km

AU      = 149_597_870.7      # km per astronomical unit
D_EM    = 384_400.0          # km, mean Earth-Moon distance

# CR3BP mass ratio: μ* = M_Moon / (M_Earth + M_Moon)
MU_STAR = MU_MOON / (MU_EARTH + MU_MOON)  # ≈ 0.012150

# CR3BP characteristic scales
_N_STAR = math.sqrt((MU_EARTH + MU_MOON) / D_EM**3)  # rad/s  ≈ 2.6653e-6
_T_STAR = 1.0 / _N_STAR                               # s      ≈ 375 190 s ≈ 4.342 d
_V_STAR = D_EM * _N_STAR                              # km/s   ≈ 1.0228 km/s

# ─────────────────────────────────────────────────────────────────────────────
# Celestial body database
# ─────────────────────────────────────────────────────────────────────────────
BODIES: Dict[str, Dict] = {
    "Earth": {"mu": MU_EARTH, "radius": R_EARTH, "soi_km":  924_000.0},
    "Moon":  {"mu": MU_MOON,  "radius": R_MOON,  "soi_km":   66_100.0},
    "Mars":  {"mu": MU_MARS,  "radius": R_MARS,  "soi_km":  577_000.0},
    "Sun":   {"mu": MU_SUN,   "radius": R_SUN,   "soi_km":  None},
}

# Simplified circular-orbit elements at J2000 (2000-Jan-1.5 TT)
# a [km], T [s], L0 [deg] mean longitude, inc [deg] to ecliptic
_EPHEM: Dict[str, Dict] = {
    "Earth": {"mu_p": MU_SUN,   "a": 1.000*AU,   "T": 365.25*86400,
              "L0": 100.466, "inc": 0.000},
    "Mars":  {"mu_p": MU_SUN,   "a": 1.524*AU,   "T": 686.97*86400,
              "L0": 355.433, "inc": 1.850},
    "Moon":  {"mu_p": MU_EARTH, "a": D_EM,        "T": 27.3217*86400,
              "L0": 218.316, "inc": 5.145},
}

# Initial Moon longitude at J2000.  The CR3BP +x axis co-rotates with the
# Moon, so the ECI→rotating angle is θ(t_s) = L₀_moon + n*·t_s.  Without
# this offset the rotating frame is misaligned by 218° at t_s=0 relative to
# SimpleEphem.moon_eci(), putting the Moon gravity in the wrong direction.
_MOON_L0_RAD = math.radians(_EPHEM["Moon"]["L0"])     # 218.316° → 3.810 rad

# ─────────────────────────────────────────────────────────────────────────────
# Simplified circular-orbit ephemeris  (no external dependencies)
# ─────────────────────────────────────────────────────────────────────────────
class SimpleEphem:
    """
    Circular-orbit position/velocity for Moon (geocentric ECI) and planets
    (heliocentric ecliptic).  Accuracy: ~1 % position error over one synodic
    period — suitable for educational / competition scenarios.

    t_s: seconds since J2000.0 (2000-Jan-01 12:00 TT).
    """

    @staticmethod
    def _state(body: str, t_s: float) -> Tuple[np.ndarray, np.ndarray]:
        e  = _EPHEM[body]
        n  = 2.0 * math.pi / e["T"]
        L  = math.radians(e["L0"]) + n * t_s
        ci = math.cos(math.radians(e["inc"]))
        si = math.sin(math.radians(e["inc"]))
        cl, sl = math.cos(L), math.sin(L)
        a  = e["a"]
        r  = np.array([a*cl, a*ci*sl, a*si*sl])
        vc = math.sqrt(e["mu_p"] / a)
        v  = vc * np.array([-sl, ci*cl, si*cl])
        return r, v

    @classmethod
    def moon_eci(cls, t_s: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Geocentric ECI position & velocity of Moon (km, km/s)."""
        return cls._state("Moon", t_s)

    @classmethod
    def earth_helio(cls, t_s: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Heliocentric ecliptic position & velocity of Earth (km, km/s)."""
        return cls._state("Earth", t_s)

    @classmethod
    def mars_helio(cls, t_s: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Heliocentric ecliptic position & velocity of Mars (km, km/s)."""
        return cls._state("Mars", t_s)

# ─────────────────────────────────────────────────────────────────────────────
# Third-body perturbation (add-on acceleration, geocentric ECI)
# ─────────────────────────────────────────────────────────────────────────────
def third_body_accel(r_sc: np.ndarray, r_body: np.ndarray,
                     mu_body: float) -> np.ndarray:
    """
    Gravitational acceleration on spacecraft at r_sc due to a third body at
    r_body (both geocentric ECI km).  Uses indirect formulation to avoid
    singularity when spacecraft approaches the body.

        a = μ_b × [ (r_b - r_sc)/|r_b - r_sc|³  −  r_b/|r_b|³ ]
    """
    d  = r_body - r_sc
    nd = _norm(d);   nb = _norm(r_body)
    return mu_body * (d / nd**3 - r_body / nb**3)


def accel_earth_moon_sun(r_sc: np.ndarray, t_s: float) -> np.ndarray:
    """
    Total ECI acceleration: Earth (two-body) + Moon + Sun (third bodies).
    No J2/drag — appropriate for cislunar free-flight phases.
    """
    r_moon, _ = SimpleEphem.moon_eci(t_s)
    r_earth_h, _ = SimpleEphem.earth_helio(t_s)
    # Sun position in geocentric ECI = −(Earth heliocentric position)
    r_sun = -r_earth_h

    a = -MU_EARTH * r_sc / _norm(r_sc)**3
    a += third_body_accel(r_sc, r_moon, MU_MOON)
    a += third_body_accel(r_sc, r_sun,  MU_SUN)
    return a

# ─────────────────────────────────────────────────────────────────────────────
# CR3BP Propagator  (Earth-Moon rotating frame, normalized units)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CR3BPState:
    """
    State vector in the Earth-Moon CR3BP normalized rotating frame.

    Coordinate origin:  Earth-Moon barycentre
    x-axis:             barycentre → Moon (co-rotating)
    z-axis:             orbital angular momentum (out of plane)

    Earth fixed at (−μ*, 0, 0);  Moon fixed at (1−μ*, 0, 0).

    Normalized scales
    -----------------
    L* = D_EM  = 384 400 km
    T* = 1/n*  ≈   4.342 days
    V* = L*n*  ≈   1.023 km/s
    """
    pos: np.ndarray   # [x, y, z]  normalized by L*
    vel: np.ndarray   # [ẋ, ẏ, ż]  normalized by V*
    tau: float = 0.0  # normalized time  τ = n* × t_s

    def jacobi(self) -> float:
        """Jacobi integral C = 2U − v²  (conserved in the CR3BP)."""
        x, y, z  = self.pos
        xd, yd, zd = self.vel
        r1 = math.sqrt((x + MU_STAR)**2 + y**2 + z**2)   # to Earth
        r2 = math.sqrt((x - 1.0 + MU_STAR)**2 + y**2 + z**2)  # to Moon
        U  = 0.5*(x**2 + y**2) + (1.0 - MU_STAR)/r1 + MU_STAR/r2
        return 2.0*U - (xd**2 + yd**2 + zd**2)

    @property
    def pos_km(self) -> np.ndarray:
        return self.pos * D_EM

    @property
    def t_days(self) -> float:
        return self.tau / (_N_STAR * 86400.0)

    @property
    def t_s(self) -> float:
        return self.tau / _N_STAR

    @property
    def speed_kms(self) -> float:
        return _norm(self.vel) * _V_STAR

    def dist_to_moon_km(self) -> float:
        x, y, z = self.pos
        return math.sqrt((x - 1.0 + MU_STAR)**2 + y**2 + z**2) * D_EM

    def dist_to_earth_km(self) -> float:
        x, y, z = self.pos
        return math.sqrt((x + MU_STAR)**2 + y**2 + z**2) * D_EM


class CR3BPPropagator:
    """
    Numerical propagator for the Earth-Moon CR3BP.

    Usage
    -----
    prop = CR3BPPropagator()
    state0 = prop.eci_to_cr3bp(r_eci_km, v_eci_kms, t_s=0.0)
    history = prop.propagate(state0, t_final_days=3.5)
    r_eci, v_eci = prop.cr3bp_to_eci(history[-1])
    """

    # ── Equations of motion ────────────────────────────────────────────────────
    @staticmethod
    def _eom(sv: np.ndarray) -> np.ndarray:
        x, y, z, xd, yd, zd = sv
        mu = MU_STAR
        r1 = math.sqrt((x + mu)**2 + y**2 + z**2)         # distance to Earth
        r2 = math.sqrt((x - 1.0 + mu)**2 + y**2 + z**2)  # distance to Moon
        c1 = (1.0 - mu) / r1**3
        c2 = mu / r2**3
        return np.array([
            xd,
            yd,
            zd,
            2.0*yd + x - c1*(x + mu)       - c2*(x - 1.0 + mu),
           -2.0*xd + y - c1*y              - c2*y,
                       - c1*z              - c2*z,
        ])

    @staticmethod
    def _rk4(sv: np.ndarray, h: float,
             eom_fn) -> np.ndarray:
        k1 = eom_fn(sv)
        k2 = eom_fn(sv + 0.5*h*k1)
        k3 = eom_fn(sv + 0.5*h*k2)
        k4 = eom_fn(sv + h*k3)
        return sv + (h/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)

    # ── Propagation ────────────────────────────────────────────────────────────
    def propagate(self, state0: CR3BPState, t_final_days: float,
                  n_steps: int = 5000) -> List[CR3BPState]:
        """
        Propagate from state0 for t_final_days physical days.
        Returns list of CR3BPState snapshots (including state0).
        Jacobi constant is conserved to ~1e-10 per step with default n_steps.
        """
        d_tau = t_final_days * 86400.0 * _N_STAR
        h = d_tau / n_steps
        sv = np.concatenate([state0.pos, state0.vel])
        tau = state0.tau
        history: List[CR3BPState] = [state0]
        for _ in range(n_steps):
            sv  = self._rk4(sv, h, self._eom)
            tau += h
            history.append(CR3BPState(sv[:3].copy(), sv[3:].copy(), tau))
        return history

    # ── Coordinate converters ──────────────────────────────────────────────────
    def eci_to_cr3bp(self, r_geo_km: np.ndarray, v_geo_kms: np.ndarray,
                     t_s: float = 0.0) -> CR3BPState:
        """
        Geocentric ECI (km, km/s) → CR3BP normalized rotating frame.

        Derivation notes
        ----------------
        The barycentre lies at μ*·L* from Earth toward Moon.
        At time t_s, the rotating frame has turned by θ = L₀_moon + n*·t_s
        from ECI, where L₀_moon = 218.316° is the Moon's mean longitude at
        J2000.  This keeps the CR3BP +x axis aligned with SimpleEphem.moon_eci().

        Position transform (inertial barycentre → rotating, normalized):
            r_norm = R(θ) × (r_geo − r_bary_eci) / L*

        Velocity transform (v_inertial = v_rotating + ω×r_rotating):
            v_rot_norm = R(θ) × (v_geo − v_bary_eci) / V*
                         − [−r_norm[1], r_norm[0], 0]
        """
        theta = _MOON_L0_RAD + _N_STAR * t_s
        ct, st = math.cos(theta), math.sin(theta)

        # Barycentre in geocentric ECI at time t_s
        r_bary = MU_STAR * D_EM * np.array([ct, st, 0.0])
        v_bary = MU_STAR * D_EM * _N_STAR * np.array([-st, ct, 0.0])

        # Translate to barycentre-centred inertial
        r_bci = r_geo_km  - r_bary
        v_bci = v_geo_kms - v_bary

        # Rotate to CR3BP frame
        R = np.array([[ct, st, 0.0], [-st, ct, 0.0], [0.0, 0.0, 1.0]])
        r_norm = (R @ r_bci) / D_EM
        # Subtract rotating-frame contribution: ω×r (ω=[0,0,1] normalized)
        omega_x_r = np.array([-r_norm[1], r_norm[0], 0.0])
        v_norm = (R @ v_bci) / _V_STAR - omega_x_r

        return CR3BPState(r_norm, v_norm, _N_STAR * t_s)

    def cr3bp_to_eci(self, state: CR3BPState) -> Tuple[np.ndarray, np.ndarray]:
        """CR3BP normalized rotating frame → geocentric ECI (km, km/s)."""
        theta = _MOON_L0_RAD + state.tau  # θ = L₀ + n*·t_s; tau = n*·t_s
        ct, st = math.cos(theta), math.sin(theta)

        # Inverse rotation (rotating → inertial barycentre)
        R_inv = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
        r_bci = R_inv @ state.pos * D_EM
        # v_inertial = v_rotating + ω×r_rotating
        omega_x_r = np.array([-state.pos[1], state.pos[0], 0.0])
        v_bci = R_inv @ (state.vel + omega_x_r) * _V_STAR

        # Translate from barycentre to geocentre
        r_bary = MU_STAR * D_EM * np.array([ct, st, 0.0])
        v_bary = MU_STAR * D_EM * _N_STAR * np.array([-st, ct, 0.0])
        return r_bci + r_bary, v_bci + v_bary

    # ── L1 / L2 Lagrange point locations ──────────────────────────────────────
    @staticmethod
    def lagrange_points() -> Dict[str, np.ndarray]:
        """
        Approximate CR3BP Lagrange point positions (normalized, on x-axis).
        L1: between Earth and Moon  (x_L1 ≈ 0.8369 for μ* = 0.01215)
        L2: behind Moon             (x_L2 ≈ 1.1557)
        L3: far side of Earth       (x_L3 ≈ −1.0051)
        L4: leading equilateral     (+60°)
        L5: trailing equilateral    (−60°)
        """
        mu = MU_STAR
        # Newton's method for L1, L2, L3
        def _find_collinear(x0, deriv_hint):
            x = x0
            for _ in range(50):
                r1 = abs(x + mu);   r2 = abs(x - 1.0 + mu)
                f  = x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
                df = 1 + (1-mu)*(-r1**2 + 3*(x+mu)**2)/r1**5 \
                       + mu*(-r2**2 + 3*(x-1+mu)**2)/r2**5
                dx = -f / (df if df != 0 else 1e-30)
                x += dx
                if abs(dx) < 1e-12:
                    break
            return np.array([x, 0.0, 0.0])

        L1 = _find_collinear(0.84, 1)
        L2 = _find_collinear(1.16, 1)
        L3 = _find_collinear(-1.0, -1)
        L4 = np.array([0.5 - mu, +math.sqrt(3)/2, 0.0])
        L5 = np.array([0.5 - mu, -math.sqrt(3)/2, 0.0])
        return {"L1": L1, "L2": L2, "L3": L3, "L4": L4, "L5": L5}

# ─────────────────────────────────────────────────────────────────────────────
# Cislunar mission planner  (patched conics, sequential segments)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CislunarResult:
    alt_leo_km: float
    alt_lmo_km: float           # lunar mission orbit altitude
    dv_tli_kms: float           # trans-lunar injection ΔV
    dv_loi_kms: float           # lunar orbit insertion ΔV
    tof_days:   float           # Earth → Moon coast time (approx)
    a_transfer_km: float        # semi-major axis of transfer ellipse
    v_inf_moon_kms: float       # hyperbolic excess speed at Moon arrival
    jacobi_approx: float        # approximate Jacobi constant of TLI trajectory

    def summary(self) -> str:
        lines = [
            "═"*52,
            "  CISLUNAR MISSION (patched conics)",
            "═"*52,
            f"  LEO altitude         : {self.alt_leo_km:.0f} km",
            f"  Lunar orbit altitude : {self.alt_lmo_km:.0f} km",
            "─"*52,
            f"  Segment 1 — Trans-Lunar Injection",
            f"    ΔV_TLI              : {self.dv_tli_kms*1000:.1f} m/s",
            f"    Transfer a          : {self.a_transfer_km/1000:.0f} Mm",
            f"    Approx TOF          : {self.tof_days:.2f} days",
            "─"*52,
            f"  Segment 2 — Lunar Orbit Insertion",
            f"    v∞ at Moon          : {self.v_inf_moon_kms*1000:.1f} m/s",
            f"    ΔV_LOI              : {self.dv_loi_kms*1000:.1f} m/s",
            "─"*52,
            f"  TOTAL ΔV             : {(self.dv_tli_kms+self.dv_loi_kms)*1000:.1f} m/s",
            f"  (Jacobi C ≈          : {self.jacobi_approx:.4f})",
            "═"*52,
        ]
        return "\n".join(lines)


class CislunarPlanner:
    """
    Patched-conics mission planner: LEO → trans-lunar injection → lunar orbit.

    Segment structure
    -----------------
    Seg-1  Earth-centred:  circular LEO → transfer ellipse  (TLI burn)
    Seg-2  Earth-centred:  coast on translunar ellipse ≈ 3 days
    Seg-3  Moon-centred:   hyperbolic approach → LOI burn → circular orbit

    The patched-conics boundary is at the Moon's sphere of influence
    (r_SOI ≈ 66 100 km from Moon centre).
    """

    def plan(self, alt_leo_km: float = 200.0,
             alt_lmo_km: float = 100.0) -> CislunarResult:
        """
        Compute ΔV budget for LEO → circular lunar orbit.

        Parameters
        ----------
        alt_leo_km : parking orbit altitude above Earth surface
        alt_lmo_km : target lunar orbit altitude above Moon surface
        """
        r_leo = R_EARTH + alt_leo_km        # geocentric radius of LEO
        mu_m  = BODIES["Moon"]["mu"]
        r_lmo = R_MOON  + alt_lmo_km        # lunar orbit radius

        # ── Segment 1: TLI burn ────────────────────────────────────────────────
        # Standard patched-conics Hohmann: apogee touches Moon's orbital radius.
        r_apogee = D_EM
        a_tl  = 0.5 * (r_leo + r_apogee)
        v_leo = math.sqrt(MU_EARTH / r_leo)
        v_tli = math.sqrt(MU_EARTH * (2.0/r_leo - 1.0/a_tl))
        dv_tli = v_tli - v_leo

        # Time of flight on transfer ellipse (half-Kepler period, approximate)
        tof_s = math.pi * math.sqrt(a_tl**3 / MU_EARTH)
        tof_d = tof_s / 86400.0

        # ── SOI boundary: speed relative to Moon ──────────────────────────────
        # Evaluate vis-viva at apogee (r = D_EM = r_apogee); r ≤ r_apogee ✓.
        # Subtract Moon's circular speed to get hyperbolic excess v∞.
        v_sc_at_apogee = math.sqrt(MU_EARTH * (2.0/r_apogee - 1.0/a_tl))
        v_moon_circ    = math.sqrt(MU_EARTH / D_EM)
        v_inf_moon     = abs(v_sc_at_apogee - v_moon_circ)

        # ── Segment 2: LOI burn ────────────────────────────────────────────────
        # Arrive at Moon SOI with v_inf, periselene = r_lmo (target)
        # Hyperbolic arrival: at periselene, speed = sqrt(v_c² + v_inf²)
        v_c_lmo   = math.sqrt(mu_m / r_lmo)
        v_hyp_peri = math.sqrt(v_c_lmo**2 + v_inf_moon**2)
        dv_loi    = v_hyp_peri - v_c_lmo

        # ── Approximate Jacobi constant of TLI trajectory ─────────────────────
        # Place spacecraft at TLI perigee (−x side in CR3BP), compute C
        prop = CR3BPPropagator()
        r_geo_tli = np.array([r_leo, 0.0, 0.0])   # perigee, Moon along +x
        v_geo_tli = np.array([0.0, v_tli, 0.0])   # prograde burn
        s0 = prop.eci_to_cr3bp(r_geo_tli, v_geo_tli, t_s=0.0)
        jacobi_c = s0.jacobi()

        return CislunarResult(
            alt_leo_km=alt_leo_km, alt_lmo_km=alt_lmo_km,
            dv_tli_kms=dv_tli,    dv_loi_kms=dv_loi,
            tof_days=tof_d,        a_transfer_km=a_tl,
            v_inf_moon_kms=v_inf_moon, jacobi_approx=jacobi_c,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Lambert solver — Izzo (2015) universal variable formulation
# Correctly handles ALL transfer angles including long-way (Δν > π)
# ─────────────────────────────────────────────────────────────────────────────
def lambert_izzo(
    r1: np.ndarray, r2: np.ndarray, tof: float,
    mu: float = MU_SUN, prograde: bool = True, nrev: int = 0,
    maxiter: int = 60, rtol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Izzo (2015) Lambert solver.  Works for all transfer angles, including
    long-way prograde transfers (Δν > π) where BLW-type solvers fail.

    Parameters
    ----------
    r1, r2   : position vectors at departure / arrival (km)
    tof      : time of flight (s) — must be > 0
    mu       : gravitational parameter of central body (km³/s²)
    prograde : True → direct / prograde orbit
    nrev     : complete extra revolutions (0 = single arc)

    Returns
    -------
    (v1, v2) : velocity vectors at r1 and r2 (km/s)

    Reference
    ---------
    Izzo, D. (2015). Revisiting Lambert's problem.
    Celest. Mech. Dyn. Astron. 121(1), 1-15.
    """
    r1m = _norm(r1)
    r2m = _norm(r2)
    c   = _norm(r2 - r1)
    s   = (r1m + r2m + c) / 2.0

    if tof <= 0.0:
        raise ValueError("lambert_izzo: TOF must be positive")
    if c < r1m * 1e-9:
        raise ValueError("lambert_izzo: degenerate geometry (r1 ≈ r2)")

    # Lambda encodes short-way (+) vs long-way (-) transfer.
    # Prograde: cross(r1,r2)·z >= 0 → dnu < π → short-way → lam > 0.
    cross_z = float(np.cross(r1, r2)[2])
    lam_abs = math.sqrt(max(0.0, 1.0 - c / s))
    if prograde:
        lam = lam_abs if cross_z >= 0.0 else -lam_abs
    else:
        lam = lam_abs if cross_z < 0.0 else -lam_abs

    # Normalized TOF: T* = tof * sqrt(2*mu / s^3)
    T_target = tof * math.sqrt(2.0 * mu / s**3)

    # Lancaster-Blanchard TOF function (Izzo 2015, eq. 9).
    # x in (-1, 1): x=+1 is parabolic prograde, x=-1 is parabolic retrograde.
    # T(x) is monotonically decreasing → large x = small TOF.
    def _tof_lb(x: float) -> float:
        x = min(max(x, -1.0 + 1e-10), 1.0 - 1e-10)
        M     = math.sqrt(1.0 - x * x)
        alpha = 2.0 * math.acos(x)
        arg   = max(-1.0, min(1.0, lam * M))
        beta  = 2.0 * math.asin(arg)
        return ((alpha - math.sin(alpha)) - (beta - math.sin(beta))
                + 2.0 * math.pi * nrev) / (2.0 * M**3)

    # Initial guess: bisect on which side of T(0) the target falls.
    T_mid = _tof_lb(0.0)
    x0 = -0.5 if T_target > T_mid else 0.6
    x0 = min(max(x0, -0.95), 0.95)

    # Halley iteration with central-difference derivatives.
    x = x0
    h = 1e-6
    for _ in range(maxiter):
        T0  = _tof_lb(x)
        Tp  = _tof_lb(x + h)
        Tm  = _tof_lb(x - h)
        dT  = (Tp - Tm) / (2.0 * h)
        d2T = (Tp - 2.0 * T0 + Tm) / h**2
        err = T0 - T_target
        if abs(err) < rtol * abs(T_target):
            break
        denom = dT * dT - 0.5 * err * d2T
        if abs(denom) > 1e-30 and abs(dT) > 1e-30:
            x -= err * dT / denom          # Halley step
        elif abs(dT) > 1e-30:
            x -= err / dT                  # Newton fallback
        x = min(max(x, -0.9999999), 0.9999999)

    # Velocity reconstruction (Izzo 2015, eq. 20).
    M     = math.sqrt(1.0 - x * x)
    y     = math.sqrt(max(0.0, 1.0 - lam * lam * M * M))
    gamma = math.sqrt(mu * s / 2.0)          # km²/s
    rho   = (r1m - r2m) / c
    sigma = math.sqrt(max(0.0, 1.0 - rho * rho))

    Vr1 = (gamma / r1m) * ((lam * y - x) - rho * (lam * y + x))
    Vr2 = -(gamma / r2m) * ((lam * y - x) + rho * (lam * y + x))
    Vt1 = (gamma / r1m) * sigma * (y + lam * x)
    Vt2 = (gamma / r2m) * sigma * (y + lam * x)

    r1hat = r1 / r1m
    r2hat = r2 / r2m
    h_vec = np.cross(r1, r2)
    h_norm = _norm(h_vec)
    if h_norm < 1e-10:
        raise ValueError("lambert_izzo: r1 and r2 are collinear")
    h_hat = h_vec / h_norm
    t1hat = np.cross(h_hat, r1hat)
    t2hat = np.cross(h_hat, r2hat)

    v1 = Vr1 * r1hat + Vt1 * t1hat
    v2 = Vr2 * r2hat + Vt2 * t2hat
    return v1, v2

# ─────────────────────────────────────────────────────────────────────────────
# Interplanetary Mission Planner (patched conics, Earth → Mars)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class InterplanetaryLeg:
    dep_body:     str
    arr_body:     str
    dep_t_s:      float          # departure time (J2000 seconds)
    arr_t_s:      float          # arrival time (J2000 seconds)
    tof_days:     float
    v_inf_dep:    np.ndarray     # departure hyperbolic excess velocity (km/s)
    v_inf_arr:    np.ndarray     # arrival hyperbolic excess velocity (km/s)
    c3_kms2:      float          # departure C3 = |v_inf_dep|² (km²/s²)
    dv_dep_kms:   float          # ΔV from parking orbit at departure
    dv_arr_kms:   float          # ΔV for capture at arrival
    total_dv_kms: float

    def summary(self) -> str:
        dep_day = self.dep_t_s / 86400.0
        lines = [
            "═"*52,
            f"  {self.dep_body} → {self.arr_body} TRANSFER (patched conics)",
            "═"*52,
            f"  Departure  J2000+{dep_day:.0f} d  ({dep_day/365.25:.2f} yr)",
            f"  TOF                 : {self.tof_days:.1f} days",
            f"  C3                  : {self.c3_kms2:.2f} km²/s²",
            f"  |v∞_dep|            : {_norm(self.v_inf_dep)*1000:.1f} m/s",
            f"  |v∞_arr|            : {_norm(self.v_inf_arr)*1000:.1f} m/s",
            "─"*52,
            f"  ΔV departure        : {self.dv_dep_kms*1000:.1f} m/s",
            f"  ΔV arrival capture  : {self.dv_arr_kms*1000:.1f} m/s",
            f"  TOTAL ΔV            : {self.total_dv_kms*1000:.1f} m/s",
            "═"*52,
        ]
        return "\n".join(lines)


class InterplanetaryPlanner:
    """
    Patched-conics Earth→Mars planner.

    Segment structure
    -----------------
    Seg-1  Earth departure:  circular parking orbit → hyperbolic escape
    Seg-2  Heliocentric:     Lambert transfer Earth → Mars (lambert_izzo)
    Seg-3  Mars arrival:     hyperbolic approach → capture burn → orbit

    Ephemeris provided by SimpleEphem (circular approximation).
    """

    def __init__(self,
                 alt_park_km: float = 200.0,
                 alt_arr_km:  float = 400.0) -> None:
        """
        Parameters
        ----------
        alt_park_km : Earth parking orbit altitude (km above surface)
        alt_arr_km  : target Mars orbit altitude (km above surface)
        """
        self.alt_park = alt_park_km
        self.alt_arr  = alt_arr_km

    def compute_leg(self, dep_t_s: float, tof_s: float) -> Optional[InterplanetaryLeg]:
        """
        Compute one Earth-Mars transfer leg.

        Parameters
        ----------
        dep_t_s : departure time in J2000 seconds
        tof_s   : time of flight in seconds

        Returns None if Lambert solver fails.
        """
        arr_t_s = dep_t_s + tof_s

        r_E, v_E = SimpleEphem.earth_helio(dep_t_s)
        r_M, v_M = SimpleEphem.mars_helio(arr_t_s)

        try:
            v1, v2 = lambert_izzo(r_E, r_M, tof_s, mu=MU_SUN, prograde=True)
        except (ValueError, RuntimeError):
            return None

        v_inf_dep = v1 - v_E
        v_inf_arr = v2 - v_M
        c3 = float(np.dot(v_inf_dep, v_inf_dep))

        # Departure ΔV: escape from circular Earth parking orbit
        r_park = R_EARTH + self.alt_park
        v_park = math.sqrt(MU_EARTH / r_park)
        v_inf_dep_mag = math.sqrt(max(0.0, c3))
        v_esc = math.sqrt(v_park**2 + c3)    # speed at periapsis of escape hyp.
        dv_dep = v_esc - v_park

        # Arrival ΔV: capture into circular Mars orbit
        r_arr  = R_MARS + self.alt_arr
        v_c_arr = math.sqrt(MU_MARS / r_arr)
        v_inf_arr_mag = _norm(v_inf_arr)
        v_hyp_arr = math.sqrt(v_c_arr**2 + v_inf_arr_mag**2)
        dv_arr = v_hyp_arr - v_c_arr

        return InterplanetaryLeg(
            dep_body="Earth", arr_body="Mars",
            dep_t_s=dep_t_s,  arr_t_s=arr_t_s,
            tof_days=tof_s/86400.0,
            v_inf_dep=v_inf_dep, v_inf_arr=v_inf_arr,
            c3_kms2=c3, dv_dep_kms=dv_dep, dv_arr_kms=dv_arr,
            total_dv_kms=dv_dep + dv_arr,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Porkchop plot (Earth → Mars launch-window grid)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PorkchopGrid:
    """Output from porkchop().  Axes are departure day and TOF (days)."""
    dep_days:    np.ndarray    # 1D array of departure J2000 days
    tof_days:    np.ndarray    # 1D array of TOFs
    dv_grid:     np.ndarray    # 2D (ndep × ntof) total ΔV [km/s]; NaN = no sol.
    c3_grid:     np.ndarray    # 2D C3 at departure [km²/s²]
    best_leg:    Optional[InterplanetaryLeg]  # lowest total ΔV


def porkchop(
    dep_j2000_days_start: float,
    dep_j2000_days_end:   float,
    dep_n:    int   = 80,
    tof_min_days: float = 130.0,
    tof_max_days: float = 350.0,
    tof_n:    int   = 60,
    alt_park_km:  float = 200.0,
    alt_arr_km:   float = 400.0,
    verbose:  bool  = True,
) -> PorkchopGrid:
    """
    Scan (departure-date × TOF) grid for Earth→Mars transfers.

    Parameters
    ----------
    dep_j2000_days_start / end : departure window in J2000 days
    dep_n                      : number of departure date samples
    tof_min_days / max_days    : TOF range to scan
    tof_n                      : number of TOF samples
    alt_park_km                : Earth parking orbit altitude
    alt_arr_km                 : Mars capture orbit altitude

    Example
    -------
    # 2026 Earth-Mars window (next optimal: ~Aug 2026)
    grid = porkchop(dep_j2000_days_start=9497, dep_j2000_days_end=9800,
                    dep_n=80, tof_min_days=130, tof_max_days=320, tof_n=60)
    """
    planner = InterplanetaryPlanner(alt_park_km=alt_park_km, alt_arr_km=alt_arr_km)

    dep_arr = np.linspace(dep_j2000_days_start, dep_j2000_days_end, dep_n)
    tof_arr = np.linspace(tof_min_days, tof_max_days, tof_n)

    dv_grid  = np.full((dep_n, tof_n), np.nan)
    c3_grid  = np.full((dep_n, tof_n), np.nan)
    best_dv  = np.inf
    best_leg: Optional[InterplanetaryLeg] = None

    for i, dep_d in enumerate(dep_arr):
        for j, tof_d in enumerate(tof_arr):
            leg = planner.compute_leg(dep_d * 86400.0, tof_d * 86400.0)
            if leg is not None:
                dv_grid[i, j] = leg.total_dv_kms
                c3_grid[i, j] = leg.c3_kms2
                if leg.total_dv_kms < best_dv:
                    best_dv = leg.total_dv_kms
                    best_leg = leg
        if verbose and (i+1) % 10 == 0:
            print(f"  porkchop: {i+1}/{dep_n} departure dates scanned…")

    if verbose:
        if best_leg is not None:
            print(f"  Best total ΔV: {best_leg.total_dv_kms*1000:.1f} m/s  "
                  f"(dep J2000+{best_leg.dep_t_s/86400:.0f}d, "
                  f"TOF {best_leg.tof_days:.0f}d)")
        else:
            print("  No valid solutions found in the scan window.")

    return PorkchopGrid(dep_days=dep_arr, tof_days=tof_arr,
                        dv_grid=dv_grid, c3_grid=c3_grid, best_leg=best_leg)

# ─────────────────────────────────────────────────────────────────────────────
# Standalone demonstration
# ─────────────────────────────────────────────────────────────────────────────
def _demo_cr3bp() -> None:
    """Propagate a translunar trajectory in the CR3BP rotating frame."""
    print("\n" + "="*60)
    print("  CR3BP DEMONSTRATION — Translunar Trajectory")
    print("="*60)

    prop = CR3BPPropagator()

    # Initial conditions: spacecraft at LEO altitude 200 km,
    # placed at the +x side (Moon initially along +x), prograde velocity
    # set for TLI (transfer ellipse with apogee ≈ D_EM).
    r_leo   = R_EARTH + 200.0          # geocentric
    a_tl    = 0.5*(r_leo + D_EM - BODIES["Moon"]["soi_km"]/2)
    v_tli   = math.sqrt(MU_EARTH*(2.0/r_leo - 1.0/a_tl))

    r_geo = np.array([r_leo, 0.0, 0.0])
    v_geo = np.array([0.0,  v_tli, 0.0])

    s0 = prop.eci_to_cr3bp(r_geo, v_geo, t_s=0.0)
    print(f"\n  Post-TLI Jacobi constant C = {s0.jacobi():.6f}")
    print(f"  Initial distance to Earth  : {s0.dist_to_earth_km():.0f} km")

    # Propagate 3.5 days
    history = prop.propagate(s0, t_final_days=3.5, n_steps=7000)

    # Find closest approach to Moon
    d_moon  = [h.dist_to_moon_km() for h in history]
    i_min   = int(np.argmin(d_moon))
    closest = history[i_min]
    print(f"  Closest Moon approach      : {d_moon[i_min]:.0f} km "
          f"at t = {closest.t_days:.2f} days")
    print(f"  Speed at closest approach  : {closest.speed_kms:.3f} km/s")

    # Verify Jacobi conservation
    C_end = history[-1].jacobi()
    print(f"  Jacobi at t=3.5d           : {C_end:.6f}  "
          f"(drift {abs(C_end - s0.jacobi())*100/abs(s0.jacobi()):.2e} %)")

    # Print Lagrange points
    lpts = prop.lagrange_points()
    print(f"\n  Lagrange points (normalized x, y):")
    for name, pt in lpts.items():
        dist_km = _norm(pt) * D_EM
        print(f"    {name}: ({pt[0]:+.4f}, {pt[1]:+.4f})  "
              f"≈ {dist_km/1000:.0f} Mm from barycentre")


def _demo_cislunar() -> None:
    print("\n" + "="*60)
    print("  CISLUNAR MISSION PLANNER")
    print("="*60)
    planner = CislunarPlanner()
    result  = planner.plan(alt_leo_km=200.0, alt_lmo_km=100.0)
    print(result.summary())


def _demo_porkchop(quick: bool = False) -> None:
    print("\n" + "="*60)
    print("  EARTH → MARS PORKCHOP PLOT SCAN")
    print("  (2026-2027 launch window  —  simplified circular ephemeris)")
    print("="*60)
    # Simplified circular ephemeris: optimal ~J2000+9816d (≈Nov 2026), TOF~205d.
    # Scan ±130 days around optimal departure, TOF 130-320d.
    grid = porkchop(
        dep_j2000_days_start=9690,   # ~Jun 2026 (early window edge)
        dep_j2000_days_end  =9950,   # ~Mar 2027 (late window edge)
        dep_n  = 20 if quick else 60,
        tof_min_days =130.0,
        tof_max_days =320.0,
        tof_n  = 15 if quick else 40,
        alt_park_km=200.0, alt_arr_km=400.0,
        verbose=True,
    )
    if grid.best_leg:
        print(grid.best_leg.summary())
    else:
        print("  [No solution found]")


def _demo_interplanetary_single() -> None:
    """Compute a single Earth-Mars leg for the ~2026 optimal window."""
    print("\n" + "="*60)
    print("  INTERPLANETARY LEG  (Earth→Mars, approx. Nov 2026 departure)")
    print("="*60)
    # Simplified ephemeris optimal: J2000+9816d (~Nov 2026), TOF=205d, C3~10.8
    dep_t_s = 9816.0 * 86400.0
    tof_s   = 205.0  * 86400.0
    planner = InterplanetaryPlanner(alt_park_km=200.0, alt_arr_km=400.0)
    leg = planner.compute_leg(dep_t_s, tof_s)
    if leg:
        print(leg.summary())
    else:
        print("  [Lambert solver did not converge for this leg]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="CR3BP / Cislunar / Interplanetary demo")
    parser.add_argument("--quick", action="store_true",
                        help="Use coarse grid for porkchop (faster demo)")
    args = parser.parse_args()

    _demo_cr3bp()
    _demo_cislunar()
    _demo_interplanetary_single()
    _demo_porkchop(quick=args.quick)

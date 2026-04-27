# 🛰️ Open Mission Ops Lab — CR3BP Cislunar & Interplanetary Simulator

> An open-source, interactive mission analysis platform covering Earth-Moon CR3BP dynamics, cislunar trajectory planning, and Earth-Mars interplanetary transfers — built for education from high school STEM to graduate-level space science.

---

## ✨ Features

### 🎓 Teaching Mode (Step-by-step Wizard)
A 3-step guided wizard for learners with no prior astrodynamics background:
- **Step 1 — CR3BP Basics**: Understand the three-body problem, rotating frame, and Lagrange point geometry
- **Step 2 — Jacobi Constant & ZVC**: Visualize zero-velocity curves and energy forbidden regions interactively
- **Step 3 — ΔV Budget**: Compute TLI + LOI delta-V for a complete cislunar mission

### 🛠 Expert Mode
Full-control interface for engineers and researchers:
- **Cislunar (Earth → Moon)**
  - CR3BP trajectory propagation (rotating + inertial frames)
  - Free-return trajectory (Artemis II–style figure-8) vs Hohmann reference overlay
  - Patched-conics TLI / LOI ΔV planner
  - Zero-velocity curve (ZVC) visualization with Lagrange-point Jacobi thresholds
  - CZML export for Cesium / NASA Worldwind visualization
- **Interplanetary (Earth → Mars)**
  - Earth–Mars porkchop plot (launch window scan)
  - Izzo (2015) Lambert solver for all transfer angles
  - Departure hyperbola C3 / ΔV and Mars hyperbolic capture (LOI) diagram

### 🔬 Physics Engine (`gmat_intercept_sim_v2_CR3BP.py`)
- Earth-Moon CR3BP propagator with RK4 integrator (Jacobi drift < 10⁻¹⁰ per step)
- Coordinate converters: geocentric ECI ↔ CR3BP normalized rotating frame (with J2000 Moon longitude offset)
- Simplified circular-orbit ephemeris — no external dependency, ~1% position accuracy over one synodic period
- Third-body perturbation (Moon + Sun) in geocentric ECI
- Izzo (2015) Lambert solver with Halley iteration
- Bisection free-return optimizer (converges target periselene in ~10 iterations)

---

## 📁 Repository Structure

```
.
├── app_cr3bp.py                     # Streamlit GUI (Teaching + Expert mode)
├── gmat_intercept_sim_v2_CR3BP.py   # Physics & mission planning engine
├── gmat_intercept_sim_v2.py         # LEO rendezvous engine (dependency)
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install streamlit numpy plotly
```

> **Python ≥ 3.9** recommended. No external astrodynamics library required.

### Launch the App

```bash
streamlit run app_cr3bp.py
```

Open your browser at `http://localhost:8501`.

### Run the Standalone Demo (CLI)

```bash
python gmat_intercept_sim_v2_CR3BP.py
# Fast porkchop scan:
python gmat_intercept_sim_v2_CR3BP.py --quick
```

---

## 🧮 Physics & Notation

| Symbol | Description | Value / Unit |
|--------|-------------|--------------|
| μ* | CR3BP mass ratio M_Moon / (M_Earth + M_Moon) | ≈ 0.012150 |
| L* | Characteristic length (Earth-Moon distance) | 384 400 km |
| T* | Characteristic time 1/n* | ≈ 4.342 days |
| V* | Characteristic velocity L* · n* | ≈ 1.023 km/s |
| C | Jacobi constant 2U − v² | conserved |

**Rotating frame origin**: Earth-Moon barycentre  
**+x axis**: barycentre → Moon (co-rotating)  
**Earth** fixed at (−μ*, 0, 0); **Moon** fixed at (1−μ*, 0, 0)

**Lagrange points** (L1–L5) computed via Newton's method; Jacobi thresholds displayed as dashed contours on the ZVC plot.

---

## 🌕 Cislunar Module

```python
from gmat_intercept_sim_v2_CR3BP import CislunarPlanner, CR3BPPropagator
import numpy as np

# Patched-conics ΔV budget
result = CislunarPlanner().plan(alt_leo_km=200, alt_lmo_km=100)
print(result.summary())

# CR3BP propagation
prop = CR3BPPropagator()
r0 = np.array([6578.0, 0.0, 0.0])   # km, LEO
v0 = np.array([0.0, 10.9, 0.0])     # km/s, post-TLI
s0 = prop.eci_to_cr3bp(r0, v0, t_s=0.0)
history = prop.propagate(s0, t_final_days=3.5)
```

---

## 🔴 Interplanetary Module (Earth → Mars)

```python
from gmat_intercept_sim_v2_CR3BP import porkchop, InterplanetaryPlanner

# Launch window scan (2026–2027)
grid = porkchop(
    dep_j2000_days_start=9690, dep_j2000_days_end=9950,
    dep_n=60, tof_min_days=130, tof_max_days=320, tof_n=40,
)
print(grid.best_leg.summary())

# Single leg
planner = InterplanetaryPlanner(alt_park_km=200, alt_arr_km=400)
leg = planner.compute_leg(dep_t_s=9816 * 86400, tof_s=205 * 86400)
print(leg.summary())
```

---

## 🗂 Mission Log & Export

- **Mission Log** (sidebar): save, compare, and delete solved trajectories within a session
- **CZML Export**: download cislunar trajectory as a `.czml` file for playback in [Cesium](https://cesium.com/cesiumjs/) or NASA Worldwind

---

## 🎯 Target Audience

| Level | Use Case |
|-------|----------|
| High school STEM | Teaching Mode wizard — intuitive step-by-step visualization |
| Undergraduate | CR3BP dynamics, Jacobi conserved quantity, Hohmann vs free-return |
| Graduate / Researcher | Expert Mode — full parameter control, patched conics, porkchop |

---

## 📐 Architecture Notes

1. **CR3BP is a separate propagator class** — does not modify the existing LEO engine (`gmat_intercept_sim_v2.py`)
2. **Patched conics uses sequential segments** — each segment has a single central body; no SOI-switching inside the integrator
3. **Lambert solver**: Izzo (2015) universal-variable formulation handles all transfer angles (Δν > π included)
4. **Ephemeris**: simplified circular-orbit model — no external dependencies; accuracy ~1% over one synodic period; clearly marked as educational only

---

## 📚 References

- Izzo, D. (2015). *Revisiting Lambert's problem*. Celestial Mechanics and Dynamical Astronomy, 121(1), 1–15.
- Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications* (4th ed.). Microcosm Press.
- Szebehely, V. (1967). *Theory of Orbits: The Restricted Problem of Three Bodies*. Academic Press.
- NASA Artemis II Mission Design Reference (public documentation)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ for open-source space education · <a href="https://github.com/">GitHub</a>
</p>

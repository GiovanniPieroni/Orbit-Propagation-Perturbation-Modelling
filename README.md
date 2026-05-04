# Space Engineering & Astrodynamics Portfolio

Welcome to my comprehensive repository of **Orbital Mechanics and Interplanetary Mission Design** projects. This portfolio showcases a variety of advanced aerospace engineering applications, ranging from high-fidelity numerical orbit propagation to optimal interplanetary transfer design, developed using **Python** and **MATLAB**.

## Key Projects

### 1. High-Fidelity Orbit Propagation & Perturbation Analysis (GONETS-M 24)
*Language: Python | Libraries: NumPy, Matplotlib, SGP4, PyYAML*

This project implements a software suite for satellite orbit propagation, validated against the **SGP4** reference model.

*   **Numerical Integration Engines**: Custom implementation of **Euler**, **Heun**, and **4th-Order Runge-Kutta (RK4)** solvers for orbital ODEs.
*   **Perturbation Modeling**:
    *   **Geopotential ($J_2$)**: Modeling Earth's non-spherical mass distribution.
    *   **Solar Radiation Pressure (SRP)**: High-fidelity model including penumbra/umbra shadowing (Doornbos algorithm).
    *   **Atmospheric Drag**: Exponential density model accounting for Earth's rotating atmosphere.
    *   **Third-Body Gravity**: Gravitational influence of the Sun and Moon using JPL Horizons ephemeris data.
*   **State Vector Optimization**: Implementation of **Grid Search** and **Gradient-based** algorithms to minimize the Cumulative Global Position Difference (CGPD) between numerical models and SGP4 data.

### 2. Interplanetary Transfer Design: Asteroid 1991VH Mission and Earth-Venus transfer
*Language: MATLAB*

A detailed study on designing optimal orbital maneuvers for a mission from a Geostationary Transfer Orbit (GTO) to a parking orbit around asteroid 1991VH.

*   **Transfer Strategies**: Comparative analysis of Bitangent, Bi-elliptic, and Plane Change maneuvers.
*   **Global Optimization**: Utilization of **Genetic Algorithms (GA)** and exhaustive Grid Searches to find launch windows and impulse sequences that minimize total $\Delta V$ and Time of Flight (TOF).
*   **Visualization**: Custom 3D plotting tools for visualizing complex orbital transfers and arrival geometries.

A study of the interplanetary transfer from Earth to Venus with various strategies:
*   **Lambert Problem Solver**: Determination of optimal trajectories between Earth and Jupiter within fixed time windows.
*   **Perturbation Benchmarking**: Evaluating the impact of planetary perturbations on the spacecraft's trajectory compared to idealized Lambert arcs, with an advanced model.
*   **Optimized transfer $\Delta$ V**: Analyzed impulsive and low-thrust maneouvres, with single- and multi-arc propagations. 
*   **SPICE Integration**: Leveraging NAIF SPICE kernels for precise planetary ephemeris and coordinate frame transformations.


### 3. JUICE Jovian System Analysis (JupitEr ICy moons Explorer) 
*Language: Python | Tools: SPICE Kernels, **TudatPy**, NumPy*

Analysis of the Jovian system dynamics for ESA's JUICE mission.
*  **Perturbation analysis**: Study of influence of other Jupiter's moons and relevant perturbations.
*  **Effect of center of propagation**: Demonstration of the influence of the center of propagation on the obtained results.  


---

##  Technical Skill Set

*   **Scientific Programming**: Python (Advanced NumPy, SciPy), MATLAB.
*   **Astrodinamics**: Numerical ODE Integration, Orbital Mechanics (Keplerian, Lambert), Perturbation Modeling (Spherical harmonics, SRP, Drag, 3rd Body).
*   **Industry Tools**: NAIF SPICE Kernels, SGP4/TLE Models, Tudat (Delft University of Technology Astrodynamics Toolbox).
*   **Data Science & Optimization**: Stochastic/Deterministic Optimization, Scientific Data Visualization, Statistical Error Analysis (GPD/CGPD).

---

## 📁 Repository Structure

```text
.
├── orbit_dynamics/
│   ├── GONETS_M24/         # Low- and Higher-fidelity Python orbit propagator
│   └── JUICE/              # JUICE perturbed orbit dynamics around Ganymede 
├── interplanetary_mission_design/
│   ├── asteroid_1991VH/    # MATLAB optimal transfer from GTO to Parking orbit about Asteroid 1991VH
│   └── Earth_venus/# Interplanetary transfer study (Python/Tudat) from Earth to Venus
└── README.md
```

---

##  How to Use

Each project directory contains its own configuration and entry-point scripts:

*   **Python Projects**: Install dependencies via `pip install numpy matplotlib pyyaml`. Run `python run_propagation.py` in the respective folder to start simulations.
*   **MATLAB Projects**: Open the `asteroid_1991VH_transfer/Code` directory in MATLAB and run `Scenario_*.m` to visualize the transfer scenarios.

---

##  Author

**Giovanni Pieroni***
*Space Engineer / Scientific Software Developer*

---
*Projects developed as part of advanced Astrodynamics and Interplanetary Mission Design curricula at TU Delft.*
* Interplanetary transfer to Asteroid 1991VH was made in collaboration with Alessandro Ponti 

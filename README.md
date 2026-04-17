# Quasi-1D Compressible Flow Solver with Heat Addition

This repository contains a Python-based numerical solver for quasi-1D, inviscid, compressible airflow in a variable-area duct (nozzle) with constant heat addition (Rayleigh-like flow). 

## Overview
The script solves the governing equations (conservation of mass, momentum, and energy) using a space-marching method coupled with a Gauss-Seidel iterative scheme. It is designed to visualize how fluid properties change across a parabolic nozzle profile when heat is applied.

## Key Features
* **Numerical Solver:** Implements Gauss-Seidel iteration to solve static temperature, pressure, density, and velocity.
* **Variable Area:** Models a parabolic nozzle geometry.
* **Heat Addition:** Simulates a constant heat transfer rate ($1000$ W/m) along the domain.
* **Shock Detection:** Includes basic checks for normal shock occurrences based on Mach number transitions.
* **Rich Visualization:** Generates standard 1D distribution plots (Temperature, Pressure, Velocity, Density, Mach Number) and maps these results onto 2D contour plots of the nozzle geometry.

## Dependencies
Ensure you have the following Python libraries installed:
* `numpy`
* `matplotlib`

## Usage
Simply clone the repository and run the script:
```bash
python quasi_1d_flow.py

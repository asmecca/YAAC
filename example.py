#!/usr/bin/env python3
"""
YAAC usage example — synthetic Monte Carlo correlator.

Generates Ncfg configurations of a two-point correlator of the form

    C(t) = A * cosh(m * (t - T/2)) + noise

then runs the full YAAC pipeline: jackknife resampling, effective mass,
constant fit, and a plot.
"""

import numpy as np
import YAAC.utils as yaac

# ------------------------------------------------------------------
# Synthetic data parameters
# ------------------------------------------------------------------
rng   = np.random.default_rng(42)
Ncfg  = 200       # number of configurations
T     = 32        # temporal extent
m     = 0.25      # true mass (lattice units)
A     = 1.0       # amplitude
noise = 0.02      # Gaussian noise level

t_arr = np.arange(T)
signal = A * np.cosh(m * (t_arr - T / 2))

# shape: (Ncfg, T)
C_raw = signal[None, :] + noise * rng.standard_normal((Ncfg, T))

# ------------------------------------------------------------------
# Build jackknife correlator
# ------------------------------------------------------------------
jk_corr = [yaac.Jackknife(C_raw[:, t]) for t in range(T)]

# ------------------------------------------------------------------
# Effective mass (cosh definition)
# ------------------------------------------------------------------
jk_meff = yaac.effective_mass(jk_corr, method="cosh")

# ------------------------------------------------------------------
# Fit effective mass to a constant in [8, 16)
# ------------------------------------------------------------------
fit_range = (8, 16)
meff_window = [jk_meff[t] for t in range(*fit_range) if jk_meff[t] is not None]
x = np.arange(*fit_range)

E, chi2 = yaac.fit_effective_mass(meff_window, fit_range=None)
print(f"Fitted mass : {yaac.format_with_error(E.mean, E.std)}")
print(f"chi2/dof    : {chi2:.3f}")
print(f"True mass   : {m}")

# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
yaac.plot_corr(jk_corr, xlabel=r"$t/a$", ylabel=r"$C(t)$",
               yscale="log", data_label="Synthetic correlator")

yaac.plot_corr(jk_meff, xlabel=r"$t/a$", ylabel=r"$am_\mathrm{eff}$",
               data_label="Cosh effective mass",
               hline=m, hlabel=f"True mass $m={m}$",
               ylim=(0.0, 0.5))

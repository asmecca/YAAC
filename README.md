# YAAC — Yet Another Analysis Code

A Python toolkit for statistical analysis of Monte Carlo data in lattice QCD and related fields. YAAC provides jackknife resampling, correlator arithmetic, effective mass computation, correlated fitting, and GEVP solving, all with full error propagation.

---

## Features

- **Jackknife resampling** — flexible `Jackknife` class supporting arbitrary estimators, binning, and bias correction
- **Correlator arithmetic** — sample-by-sample addition, subtraction, multiplication, division, and power for both single observables and full correlators
- **Effective mass** — log and cosh definitions with automatic initial-guess generation
- **Correlated fits** — jackknife bin-by-bin fitting via `scipy.optimize.curve_fit` with full covariance matrix support
- **GEVP** — generalised eigenvalue problem solver with Cholesky decomposition, eigenvalue and eigenvector sorting, and jackknife error propagation
- **Plotting** — `errorbar`-based correlator and multi-correlator plots with optional log scale, reference lines, and save-to-file

---

## Installation

**Requirements:** Python ≥ 3.7, NumPy, SciPy, Matplotlib.

Clone the repository and install in editable mode:

```bash
git clone https://github.com/<your-username>/YAAC.git
cd YAAC
pip install -e .
```

Or run the provided helper script from the repo root:

```bash
bash install.sh
```

---

## Quick start

```python
import numpy as np
import YAAC.utils as yaac

# --- Build a jackknife object from raw Monte Carlo samples ---
rng = np.random.default_rng(0)
raw = rng.standard_normal(200)          # 200 configurations
jk  = yaac.Jackknife(raw)

print(f"mean = {jk.mean:.4f} ± {jk.std:.4f}")

# --- Arithmetic between jackknife objects ---
jk2 = yaac.Jackknife(rng.standard_normal(200) + 1.0)
jk_sum  = yaac.jack_add(jk, jk2)
jk_prod = yaac.jack_mul(jk, jk2)

# --- Read a correlator from file and compute effective mass ---
# jk_corr = yaac.read_corr("correlator.dat")
# jk_meff  = yaac.effective_mass(jk_corr, method="cosh")
# yaac.plot_corr(jk_meff, xlabel=r"$t/a$", ylabel=r"$am_\mathrm{eff}$")
```

A fully worked example using synthetic data is provided in [`example.py`](example.py).

---

## API overview

### `Jackknife` class

```python
Jackknife(data, estimator=np.mean, binsize=None, nbins=None)
```

| Attribute | Description |
|---|---|
| `jk.mean` | Jackknife mean |
| `jk.std` | Jackknife standard deviation |
| `jk.var` | Jackknife variance |
| `jk.theta` | Full-sample estimator |
| `jk.unbiased` | Bias-corrected estimator |
| `jk.jk_samples` | Array of leave-one-out samples |
| `jk.N` | Number of jackknife bins |

**Class methods:**

- `Jackknife.from_samples(jk_samples)` — construct directly from pre-computed samples (used for error propagation)

### I/O

| Function | Description |
|---|---|
| `read_corr(filename, ...)` | Read a two-point correlator from file; returns `list[Jackknife]` |

File format: first line must be `Ncfg  N_t  N_t/2`, followed by one value per line.

### Arithmetic

Single observables:

| Function | Operation |
|---|---|
| `jack_add(jk1, jk2)` | `jk1 + jk2` |
| `jack_add_d(jk1, d)` | `jk1 + d` |
| `jack_mul(jk1, jk2)` | `jk1 * jk2` |
| `jack_mul_d(jk1, d)` | `jk1 * d` |
| `jack_div(jk1, jk2)` | `jk1 / jk2` |
| `jack_div_d(jk1, d)` | `jk1 / d` |
| `jack_pow(jk1, d)` | `jk1 ** d` |
| `jack_exp(jk1)` | `exp(jk1)` |

Correlator wrappers (operate time-slice by time-slice):
`add_corrs`, `add_corrs_d`, `multiply_corrs`, `multiply_corr_d`, `divide_corrs`, `divide_corr_d`, `corr_pow`

### Statistics

| Function | Description |
|---|---|
| `jackknife_covariance(jk)` | Variance (single `Jackknife`) or covariance matrix (list) |
| `bin_data(data, binsize=None, nbins=None)` | Block-average raw data along an axis |

### Effective mass

| Function | Description |
|---|---|
| `effective_mass(jack_C, method="cosh")` | Compute effective mass; supports `"log"` and `"cosh"` |
| `fit_effective_mass(jack_C, fit_range=None)` | Fit effective mass to a constant; returns `(E, chi2/dof)` |

### Fitting

```python
params, chi2_red = jackknife_fit(corrs, x, fit_func, p0,
                                  fit_range=None, correlated=False,
                                  cov=None, absolute_sigma=True)
```

Performs bin-by-bin jackknife fitting. Returns an array of shape `(Njack, Nparams)` and the median reduced χ².

### GEVP

```python
lambdas, vecs = gevp(G, t0, ts=None,
                     sort="Eigenvalue", vector_obs=True, symmetrise=True)
```

Solves G(t) v = λ G(t₀) v for a correlator matrix `G[t][i][j]` of `Jackknife` objects. Supports eigenvalue sorting, eigenvector tracking (`sort="Eigenvector"`), and full jackknife error propagation on both eigenvalues and eigenvectors.

```python
Cproj = project_state(G, vecs, state=0)
```

Projects the correlator matrix onto a single state using the GEVP eigenvectors.

### Plotting

| Function | Description |
|---|---|
| `plot_corr(corr, xlabel, ylabel, ...)` | Plot a single jackknifed correlator |
| `plot_multi_corr(list_corr, xlabel, ylabel, ...)` | Overlay multiple correlators |

Both functions accept `ylim`, `yscale`, `hline`/`hlabel`, `vline`/`vlabel`, and `save` arguments.

### Utilities

| Function | Description |
|---|---|
| `format_with_error(value, error, nsig=2)` | Format as `x.xxx(yy)` |

---

## Custom plot style

YAAC will automatically load `myplot.mplstyle` from the package directory if it is present. If the file is absent, matplotlib's default style is used. To use your own style, place a `.mplstyle` file named `myplot.mplstyle` in the `YAAC/` package directory.

---

## License

MIT — see [LICENSE](LICENSE).

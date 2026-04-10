#!/usr/bin/env python3
import itertools
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

_MPLSTYLE = os.path.join(os.path.dirname(__file__), 'myplot.mplstyle')
if os.path.isfile(_MPLSTYLE):
    plt.style.use(_MPLSTYLE)

class Jackknife:
    
    def __init__(self, data, estimator=lambda x: np.mean(x), binsize=None, nbins=None):
        """
        Parameters
        ----------
        data : array-like, shape (Ncfg, ...)
            Raw Monte Carlo data
        estimator : callable
            Function mapping data -> observable
        """
        self.data = np.asarray(data)
        self.estimator = estimator
        if binsize is not None or nbins is not None:
            self.data = bin_data(self.data, binsize=binsize, nbins=nbins, axis=0)
        self.N = self.data.shape[0]
        self.binsize = binsize
        self.nbins_requested = nbins

        self._compute()

    # --------------------------------------------------
    def _compute(self):
        # Full-sample estimator
        self.theta = self.estimator(self.data)

        # Leave-one-out jackknife samples
        jk_data = self._leave_one_out(self.data)
        self.jk_samples = np.array([self.estimator(d) for d in jk_data])

        # Jackknife mean
        self.mean = np.mean(self.jk_samples, axis=0)

        # Jackknife variance (unbiased)
        diff = self.jk_samples - self.theta
        self.var = (self.N - 1) / self.N * np.sum(diff**2, axis=0)
        self.std = np.sqrt(self.var)

        # Bias-corrected estimator
        self.unbiased = self.N * self.theta - (self.N - 1) * self.mean

    # --------------------------------------------------
    @staticmethod
    def _leave_one_out(data):
        """
        Efficient leave-one-out views.
        """
        N = data.shape[0]
        return [np.delete(data, i, axis=0) for i in range(N)]

    @classmethod
    def from_samples(cls, jk_samples, theta=None):
        obj = cls.__new__(cls)
        obj.jk_samples = np.asarray(jk_samples)
        obj.N = obj.jk_samples.shape[0]
        obj.mean = np.mean(obj.jk_samples, axis=0)
        obj.theta = theta if theta is not None else obj.mean 
        diff = obj.jk_samples - obj.theta
        obj.var = (obj.N - 1) / obj.N * np.sum(diff**2, axis=0)
        obj.std = np.sqrt(obj.var)
        obj.unbiased = obj.N * obj.theta - (obj.N - 1) * obj.mean
        obj.data = None
        obj.estimator = None
        return obj

def _get_jackknife_samples(jk):
    return np.asarray(jk.jk_samples)

def bin_data(data, binsize=None, nbins=None, axis=0, drop_remainder=True):
    """
    Bin/block raw data along one axis by averaging inside each bin.
    """
    x = np.asarray(data)
    x = np.moveaxis(x, axis, 0)

    N = x.shape[0]

    if (binsize is None) == (nbins is None):
        raise ValueError("Give exactly one of binsize or nbins")

    if binsize is not None:
        if not isinstance(binsize, int) or binsize <= 0:
            raise ValueError("binsize must be a positive integer")
        nbins_eff = N // binsize
        if nbins_eff < 1:
            raise ValueError("Not enough data for one full bin")
        Nused = nbins_eff * binsize
    else:
        if not isinstance(nbins, int) or nbins <= 0:
            raise ValueError("nbins must be a positive integer")
        if nbins > N:
            raise ValueError("nbins cannot be larger than the data length")
        binsize = N // nbins
        if binsize < 1:
            raise ValueError("Computed binsize is smaller than 1")
        nbins_eff = nbins
        Nused = nbins_eff * binsize

    if Nused != N and not drop_remainder:
        raise ValueError("Data size is not divisible into full bins")

    x = x[:Nused]
    x = x.reshape((nbins_eff, binsize) + x.shape[1:])
    x = np.mean(x, axis=1)

    return np.moveaxis(x, 0, axis)

def read_corr(filename, tempo=None, from_samples=False, col2=False,
              binsize=None, nbins=None):
    """
    Reading Correlator:
      Data file must have the following first line:
      Ncnfg N_t N_t/2
    """
    corr = []

    with open(filename, 'r') as f:
        first_line = f.readline().strip('\n')
        x = first_line.split()
        cnfg = int(x[0])
        time = int(x[1])
        T = int(x[2])

        for line in f:
            x = line.split()
            if x[0] != str(cnfg):
                corr += [float(x[1])] if col2 else [float(x[0])]

    if tempo is not None:
        time = tempo

    C = np.zeros((cnfg, time), dtype=float)
    for i in range(cnfg):
        for t in range(time):
            C[i][t] = corr[i * time + t]

    if (binsize is not None) or (nbins is not None):
        C = bin_data(C, binsize=binsize, nbins=nbins, axis=0)

    t_C = C.T
    jk_corr = [None] * time
    if from_samples:
        for t in range(time):
            jk_corr[t] = Jackknife.from_samples(t_C[t])
    else:
        for t in range(time):
            jk_corr[t] = Jackknife(t_C[t])
    return jk_corr

def symmetrise(corr):
    T = int(len(corr))
    if T % 2 != 0:
        raise ValueError("corr is not even")
    for t in range(1, T // 2):
        sym = jack_mul_d(jack_add(corr[t], corr[T - t]), 0.5)
        corr[t] = sym
        corr[T - t] = sym
    return corr

def plot_corr(corr, xlabel, ylabel, ylim=None, yscale=None, data_label=None, color='blue', marker='o', ncol=1, save=None, hline=None, hlabel=None, vline=None, vlabel=None):
    # plots jackknife correlator
    for t in range(len(corr)):
        if corr[t] is not None:
            label = data_label if t == 0 else None
            plt.errorbar(x=t, y=corr[t].mean, yerr=corr[t].std, color=color, fmt=marker, label=label)
    if data_label is not None:
        plt.legend(loc='best', ncol=ncol)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if hline is not None:
        plt.axhline(y=hline, color='black', ls='--', label=hlabel)
    if vline is not None:
        plt.axvline(x=vline, color='black', ls='--', label=vlabel)
    if ylim is not None:
        y_i, y_f = ylim
        plt.ylim(y_i,y_f)    
    if yscale is not None:
        plt.yscale(yscale)
    if save is not None:
        fig=plt.gcf()
        fig.savefig(save)
    plt.show()

def plot_multi_corr(list_corr,xlabel,ylabel,xlim=None,ylim=None,yscale=None,list_label=None,ncol=1,save=None,x_offset=None,hline=None,herr=None,hlabel=None,vline=None,verr=None,vlabel=None):
    # plots many correlators for comparisons
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if hline is not None:
        plt.axhline(y=hline,color='black',ls='--',label=hlabel)
    if herr is not None:
        plt.axhspan(hline-herr,hline+herr,color='gray',alpha=0.4)    
    if vline is not None:
        plt.axvline(x=vline,color='black',ls='--',label=vlabel)
    if verr is not None:
        plt.axvspan(vline-verr,vline+verr,color='gray',alpha=0.4)
    if ylim is not None:
        y_i, y_f = ylim
        plt.ylim(y_i,y_f)
    if xlim is not None:
        x_i, x_f = xlim
        plt.xlim(x_i,x_f)
    if yscale is not None:
        plt.yscale(yscale)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 's', '^', 'v', 'D', '*', 'P', 'X']
    for i in range(len(list_corr)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]        
        corr = list_corr[i]
        offset=0.0
        if x_offset is not None:
            offset=x_offset
        if list_label is not None:
            data_label = list_label[i]
        else:
            data_label = None
        for t in range(len(corr)):
            if corr[t] is not None and corr[t] is not np.nan:
                label = data_label if t == 0 else None
                plt.errorbar(x=t + i * offset, y=corr[t].mean, yerr=corr[t].std, color=color, fmt=marker, label=label)
        if data_label is not None:
            plt.legend(loc='best', ncol=ncol)
    if save is not None:
        fig=plt.gcf()
        fig.savefig(save)
    plt.show()
    

def jackknife_covariance(jk):
    """
    Compute jackknife variance / covariance.

    Parameters
    ----------
    jk : Jackknife or sequence of Jackknife
        Single Jackknife object or list/array of Jackknife objects
        (e.g. one per time-slice)

    Returns
    -------
    cov : ndarray
        - single Jackknife  -> scalar or array (variance)
        - list of Jackknife -> 2D covariance matrix
    """

    # --------------------------------------------------
    # Case 1: single Jackknife object
    # --------------------------------------------------
    if isinstance(jk, Jackknife):
        samples = jk.jk_samples
        mean = jk.mean
        N = jk.N

        diff = samples - jk.theta
        return (N - 1) / N * np.sum(diff * diff, axis=0)

    # --------------------------------------------------
    # Case 2: array / list of Jackknife objects
    # --------------------------------------------------
    jk_list = list(jk)
    Nt = len(jk_list)

    # Consistency check
    N = jk_list[0].N
    for j in jk_list:
        if j.N != N:
            raise ValueError("All Jackknife objects must have the same N")

    # Stack jackknife samples: shape (Njack, Nt, ...)
    samples = np.stack([j.jk_samples for j in jk_list], axis=1)

    # Mean over jackknife samples
    mean = np.mean(samples, axis=0)

    # Reshape to (Njack, Nt) for correlator-like objects
    theta = np.array([j.theta for j in jk_list])  # shape (Nt,)
    diff = samples - theta  # broadcasts: (Njack, Nt) - (Nt,)


    # Covariance: (Nt x Nt)
    cov = (N - 1) / N * np.tensordot(diff, diff, axes=(0, 0))

    return cov

def jack_add(jk1, jk2):
    """
    Add two Jackknife objects sample-by-sample.

    Parameters
    ----------
    jk1, jk2 : Jackknife
        Objects with identical N and compatible shapes

    Returns
    -------
    Jackknife
        New jackknifed object representing the sum
    """
    if jk1.N != jk2.N:
        raise ValueError("Jackknife objects must have the same N")

    samples = jk1.jk_samples + jk2.jk_samples
    return Jackknife.from_samples(samples, theta=jk1.theta + jk2.theta)


def add_corrs(corr1, corr2):
    """
    Add two Jackknife correlators sample-by-sample.

    Parameters
    ----------
    corr1, corr2 : list of Jackknife

    Returns
    -------
    list of Jackknife
        New jackknifed correlator representing the sum
    """
    if len(corr1) != len(corr2):
        raise ValueError("Correlators must have the same length")

    res = [None] * len(corr1)
    for t in range(len(corr1)):
        res[t] = jack_add(corr1[t], corr2[t])

    return res

def jack_add_d(jk1, d):
    """
    Add a scalar to a Jackknife object sample-by-sample.

    Parameters
    ----------
    jk1 : Jackknife
    d : scalar

    Returns
    -------
    Jackknife
        New jackknifed object representing the sum
    """
    samples = jk1.jk_samples + d
    return Jackknife.from_samples(samples, theta=jk1.theta + d)


def add_corrs_d(corr1, d):
    """
    Add a scalar to a Jackknife correlator sample-by-sample.

    Parameters
    ----------
    corr1 : list of Jackknife
    d : scalar

    Returns
    -------
    list of Jackknife
        New jackknifed correlator representing the sum
    """
    res = [None] * len(corr1)
    for t in range(len(corr1)):
        res[t] = jack_add_d(corr1[t], d)

    return res

def jack_mul(jk1, jk2):
    """
    Multiply two Jackknife objects sample-by-sample.

    Parameters
    ----------
    jk1, jk2 : Jackknife
        Objects with identical N and compatible shapes

    Returns
    -------
    Jackknife
        New jackknifed object representing the product
    """
    if jk1.N != jk2.N:
        raise ValueError("Jackknife objects must have the same N")

    samples = jk1.jk_samples * jk2.jk_samples
    return Jackknife.from_samples(samples, theta=jk1.theta * jk2.theta)

def jack_mul_d(jk1, d):
    """
    Multiply a Jackknife object by a scalar sample-by-sample.

    Parameters
    ----------
    jk1 : Jackknife
    d : scalar (int, float, or numpy scalar)

    Returns
    -------
    Jackknife
        New jackknifed object representing the product
    """
    if not np.isscalar(d):
        raise ValueError("d must be a scalar")

    samples = jk1.jk_samples * d
    return Jackknife.from_samples(samples, theta=jk1.theta * d)

def jack_div_d(jk1, d):
    """
    Divide a Jackknife object by a scalar sample-by-sample.

    Parameters
    ----------
    jk1 : Jackknife
    d : scalar (int, float, or numpy scalar)

    Returns
    -------
    Jackknife
        New jackknifed object representing the quotient
    """
    if not np.isscalar(d):
        raise ValueError("d must be a scalar")

    samples = jk1.jk_samples / d
    return Jackknife.from_samples(samples, theta=jk1.theta / d)

def multiply_corrs(corr1, corr2):
    """
    Multiply two Jackknife correlators sample-by-sample.

    Parameters
    ----------
    corr1, corr2 : list of Jackknife

    Returns
    -------
    list of Jackknife
        New jackknifed correlator representing the product
    """
    if len(corr1) != len(corr2):
        raise ValueError("Correlators must have the same length")

    res = [None] * len(corr1)
    for t in range(len(corr1)):
        res[t] = jack_mul(corr1[t], corr2[t])

    return res

def multiply_corr_d(corr1, d):
    """
    Multiply a Jackknife correlator by a scalar sample-by-sample.

    Parameters
    ----------
    corr1 : list of Jackknife
    d : scalar (int, float, or numpy scalar)

    Returns
    -------
    list of Jackknife
        New jackknifed correlator representing the product
    """
    if not np.isscalar(d):
        raise ValueError("d must be a scalar")

    res = [None] * len(corr1)
    for t in range(len(corr1)):
        res[t] = jack_mul_d(corr1[t], d)

    return res

def jack_div(jk1, jk2, check_zero=True):
    """
    Divide two Jackknife objects sample-by-sample.

    Parameters
    ----------
    jk1, jk2 : Jackknife
        Objects with identical N and compatible shapes
    check_zero : bool
        If True (default), samples with a zero denominator are set to NaN
        instead of raising an exception.

    Returns
    -------
    Jackknife
        New jackknifed object representing the quotient.
        Samples whose denominator is zero are set to NaN.
    """
    if jk1.N != jk2.N:
        raise ValueError("Jackknife objects must have the same N")

    denom = jk2.jk_samples

    with np.errstate(divide='ignore', invalid='ignore'):
        samples = np.where(denom == 0.0, np.nan, jk1.jk_samples / denom)
    theta = jk1.theta / jk2.theta if jk2.theta != 0.0 else np.nan
    return Jackknife.from_samples(samples, theta=theta)

def divide_corrs(corr1, corr2):
    """
    Divide two Jackknife correlators sample-by-sample.

    Parameters
    ----------
    corr1, corr2 : list of Jackknife

    Returns
    -------
    list of Jackknife
        New jackknifed correlator representing the quotient
    """
    if len(corr1) != len(corr2):
        raise ValueError("Correlators must have the same length")

    res = [None] * len(corr1)
    for t in range(len(corr1)):
        res[t] = jack_div(corr1[t], corr2[t])

    return res

def divide_corr_d(corr1, d):
    """
    Divide a Jackknife correlator by a scalar sample-by-sample.

    Parameters
    ----------
    corr1 : list of Jackknife
    d : scalar (int, float, or numpy scalar)

    Returns
    -------
    list of Jackknife
        New jackknifed correlator representing the quotient
    """
    if not np.isscalar(d):
        raise ValueError("d must be a scalar")

    res = [None] * len(corr1)
    for t in range(len(corr1)):
        res[t] = jack_div_d(corr1[t], d)

    return res

def jack_exp(jk1):
    """
    Apply exp to a Jackknife object sample-by-sample.

    Parameters
    ----------
    jk1 : Jackknife

    Returns
    -------
    Jackknife
        New jackknifed object representing exp(jk1)
    """
    samples = np.exp(jk1.jk_samples)
    return Jackknife.from_samples(samples, theta=np.exp(jk1.theta))

def jack_pow(jk1, d):
    """
    Raise a Jackknife object to a power sample-by-sample.

    Parameters
    ----------
    jk1 : Jackknife
    d : scalar

    Returns
    -------
    Jackknife
        New jackknifed object representing jk1 ** d
    """
    samples = np.power(jk1.jk_samples, d)
    return Jackknife.from_samples(samples, theta=np.power(jk1.theta, d))


def corr_pow(corr, d):
    """
    Raise a Jackknife correlator to a power sample-by-sample.

    Parameters
    ----------
    corr : list of Jackknife
    d : scalar

    Returns
    -------
    list of Jackknife
        New jackknifed correlator representing corr ** d
    """
    res = [None] * len(corr)
    for t in range(len(corr)):
        res[t] = jack_pow(corr[t], d)

    return res

def find_root_newton(d, root_function, guess, tol=1e-12, maxiter=100): #1e-12
    """
    Simple Newton–secant root finder for scalar equations.

    Parameters
    ----------
    d : float
        Data parameter passed to root_function
    root_function : callable
        f(x, d)
    guess : float
        Initial guess
    """

    x = guess

    for _ in range(maxiter):
        fx = root_function(x, d)

        # Finite-difference derivative
        h = 1e-6
        dfx = (root_function(x + h, d) - fx) / h

        if dfx == 0:
            break

        x_new = x - fx / dfx

        if np.abs(x_new - x) < tol:
            return x_new

        x = x_new

    # If it didn't converge, return NaN
    return np.nan


def meff_cosh_from_ratio(R, t, T, guess):
    """
    Solve cosh effective mass equation using a guess-based solver.
    """

    x = t - T / 2
    y = t + 1 - T / 2

    def root_function(m, d):
        return np.cosh(m * x) / np.cosh(m * y) - d

    m = find_root_newton(R, root_function, guess)
    return np.abs(m)

def log_meff_guess(jack_C_t, jack_C_tp1):
    """
    Central-value log effective mass used as initial guess.
    """
    if jack_C_t.mean <= 0 or jack_C_tp1.mean <= 0:
        return None

    return np.log(jack_C_t.mean / jack_C_tp1.mean)

def effective_mass(jack_C, method="cosh"):
    """
    Compute the effective mass from a jackknifed correlator.

    Parameters
    ----------
    jack_C : sequence of Jackknife
        Correlator as an array/list of Jackknife objects (length Nt)
    method : str
        Effective mass definition.
        Currently supported:
        - "cosh": solves C(t)/C(t+1) = cosh(m * (t-T/2)) / cosh(m * (t+1 - T/2))
        - "log" : log(C(t) / C(t+1))

    Returns
    -------
    jack_meff : list of Jackknife
        Effective mass as an array of Jackknife objects (length Nt-1)
    """

    jack_C = list(jack_C)
    Nt = len(jack_C)

    jack_meff = []
    
    for t in range(Nt - 1):
        # Ratio C(t) / C(t+1); zero denominator samples become NaN
        jk_ratio = jack_div(jack_C[t], jack_C[t + 1])

        if method == "log":
            # Apply log sample-by-sample; non-positive ratios (incl. NaN) -> NaN
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_samples = jk_ratio.jk_samples
                samples = np.where(ratio_samples > 0,
                                   np.log(ratio_samples), np.nan)

            # If every sample is NaN the timeslice is unusable
            if np.all(np.isnan(samples)):
                jack_meff.append(None)
                continue
            
            theta = np.log(jk_ratio.theta) if jk_ratio.theta > 0 else np.nan

        elif method == "cosh":
            guess = log_meff_guess(jack_C[t], jack_C[t + 1])
            if guess is not None:
                guess = abs(guess)

            if guess is None or not np.isfinite(guess):
                jack_meff.append(None)
                continue
            
            samples = []

            for R in jk_ratio.jk_samples:
                if not np.isfinite(R) or R <= 0:
                    samples.append(np.nan)
                else:
                    samples.append(
                        meff_cosh_from_ratio(R, t, Nt, guess)
                    )

            samples = np.array(samples)

            # If every sample is NaN the timeslice is unusable
            if np.all(np.isnan(samples)):
                jack_meff.append(None)
                continue
            
            theta = meff_cosh_from_ratio(jk_ratio.theta, t, Nt, guess) if (  
                np.isfinite(jk_ratio.theta) and jk_ratio.theta > 0) else np.nan
            
        else:
            raise NotImplementedError(
                f"Effective mass method '{method}' not implemented"
            )

        # Build Jackknife object from derived samples
        jk_meff = Jackknife.from_samples(samples, theta=theta)
        jack_meff.append(jk_meff)

    return jack_meff

def fit_effective_mass(jack_C, fit_range=None):
    """
    fits effective mass jackknife correlator to a constant
    
    Returns
    ---------
        E: fitted energy
        chi2: correlated chi^2/d.o.f.    
    """
    
    def cnst_func(x, a):
        return a

    x = np.arange(len(jack_C))
    cov = jackknife_covariance(jack_C)
    params, chi2 = jackknife_fit(jack_C, x, cnst_func, p0=[1.0], fit_range=fit_range, cov=cov, correlated=True)

    # Fit on full-sample thetas to get theta for the output Jackknife  ← new
    tmin, tmax = fit_range if fit_range is not None else (0, len(jack_C))
    y_full = np.array([jk.theta for jk in jack_C[tmin:tmax]])
    x_fit = x[tmin:tmax]
    popt_full, _ = curve_fit(cnst_func, x_fit, y_full, p0=[1.0])  # ← new

    E = Jackknife.from_samples(params[:, 0], theta=popt_full[0])  # ← theta added
    return E, chi2
    

def jackknife_fit(corrs, x, fit_func, p0, fit_range=None, correlated=False, cov=None, absolute_sigma=True):
    """
    Jackknife bin-by-bin fitting routine.

    Parameters
    ----------
    corrs : list
        List of jackknife objects. Each object must expose an array
        of samples with shape (Njack, Nt).
    x : array_like
        x-values corresponding to data points.
    fit_func : callable
        Fit function f(x, *params).
    p0 : array_like
        Initial guess for fit parameters.
    fit_range : tuple or None
        (tmin, tmax) indices. If None, use full range.
    correlated : bool
        If True, perform correlated fits.
    cov : ndarray or None
        Covariance matrix (Nt x Nt). Required if correlated=True.
    absolute_sigma : bool
        Passed to curve_fit.

    Returns
    -------
    fit_params : list of ndarray
        List of arrays of shape (Njack, Nparams), one per correlator.
    chi2_red_median : list of float
        Median reduced chi^2 for each correlator.
    """
    # ---- corrs = list of timeslices ----
    if not isinstance(corrs, (list, tuple)):
        raise TypeError("corrs must be a list of Jackknife objects (one per timeslice)")

    Nt = len(corrs)
    if Nt == 0:
        raise ValueError("Empty correlator list")

    Nj = corrs[0].N

    data_all = np.zeros((Nj, Nt))
    for t, jk in enumerate(corrs):
        if jk.N != Nj:
            raise ValueError("Inconsistent number of jackknife bins")
        data_all[:, t] = jk.jk_samples

    # Fit window
    if fit_range is not None:
        tmin, tmax = fit_range
        data_all = data_all[:, tmin:tmax]
        x_fit = x[tmin:tmax]
    else:
        tmin, tmax = 0, Nt
        x_fit = x

    Nj, Nt_fit = data_all.shape
    npar = len(p0)
    dof = Nt_fit - npar
    if dof <= 0:
        raise ValueError("Non-positive degrees of freedom")
    

    params_jack = np.zeros((Nj, npar))
    chi2_red = np.zeros(Nj)

    # Covariance handling
    if correlated:
        if cov is None:
            raise ValueError("Covariance matrix must be provided for correlated fits.")
        cov_fit = cov[tmin:tmax, tmin:tmax]
        #cov_fit += 1e-12 * np.eye(Nt)
        L = np.linalg.cholesky(cov_fit)

    for i in range(Nj):
        y = data_all[i]

        if correlated:
            popt, _ = curve_fit(
                fit_func,
                x_fit,
                y,
                p0=p0,
                sigma=cov_fit,
                absolute_sigma=absolute_sigma
            )
            r = y - fit_func(x_fit, *popt)
            ychi = np.linalg.solve(L, r)
            chi2 = np.dot(ychi, ychi)
        else:
            popt, _ = curve_fit(
                fit_func,
                x_fit,
                y,
                p0=p0
            )
            r = y - fit_func(x_fit, *popt)
            chi2 = np.sum(r**2)

        params_jack[i] = popt
        chi2_red[i] = chi2 / dof

    return params_jack, np.median(chi2_red)
    

def format_with_error(value, error, nsig=2):
    """
    Format a value with uncertainty as x.xxx(yy).
 
    The parenthetical notation x.xxx(yy) means the last displayed digits
    of the value are uncertain by yy. Works correctly for both sub-unit
    errors (e.g. 1.234(25)) and multi-digit errors (e.g. 320.5(25),
    1230(120)).
 
    Parameters
    ----------
    value : float
    error : float
    nsig : int
        Number of significant digits for the error (default 2).
    """
    if error <= 0:
        return f"{value}"
 
    # Order of magnitude of the error
    exp = int(np.floor(np.log10(error)))
 
    if exp < 0:
        # Error is smaller than 1: use decimal notation
        decimals = -(exp - (nsig - 1))
        err_rounded = round(error, -exp + (nsig - 1))
        val_rounded = round(value, decimals)
        err_int = int(round(err_rounded * 10**decimals))
        fmt = f"{{:.{decimals}f}}({{}})"
        return fmt.format(val_rounded, err_int)
    else:
        # Error is >= 1: round both to the same significant place
        round_to = exp - (nsig - 1)
        if round_to >= 0:
            # Error rounds to tens, hundreds, etc.
            scale = 10**round_to
            err_int = int(round(error / scale)) * scale
            val_int = int(round(value / scale)) * scale
            return f"{val_int}({err_int})"
        else:
            # Error >= 1 but rounds to a decimal place (e.g. 2.5 with nsig=2)
            decimals = -round_to
            err_rounded = round(error, decimals)
            val_rounded = round(value, decimals)
            err_int = int(round(err_rounded * 10**decimals))
            fmt = f"{{:.{decimals}f}}({{}})"
            return fmt.format(val_rounded, err_int)
        
def _symmetrise(M):
    return 0.5 * (M + np.swapaxes(M, -1, -2))

def _gevp_one_cholesky(Gt, G0):
    """
    Solve Gt v = lam G0 v for numeric matrices (N,N).
    Returns:
      w : (N,) eigenvalues in descending order
      v : (N,N) eigenvectors as columns, normalised so v^T G0 v = 1
    """
    # Cholesky of G0
    L = np.linalg.cholesky(G0)
    Linv = np.linalg.inv(L)

    # Convert to standard EVP: C u = lam u  with C = Linv Gt Linv^T
    C = Linv @ Gt @ Linv.T
    C = 0.5 * (C + C.T)  # clean numerical asymmetry

    w, u = np.linalg.eigh(C)  # ascending
    idx = np.argsort(w)[::-1]  # descending
    w = w[idx]
    u = u[:, idx]

    # Back-transform eigenvectors
    v = Linv.T @ u

    # Normalise with respect to G0: v^T G0 v = 1
    for k in range(v.shape[1]):
        nrm2 = v[:, k].T @ G0 @ v[:, k]
        nrm = np.sqrt(nrm2) if nrm2 > 0 else 1.0
        v[:, k] /= nrm

    return w, v

def _best_permutation_by_overlap(v_ref, v_t):
    """
    Find permutation p that maximises sum_i |<v_ref_i | v_t_{p(i)}>| in Euclidean inner product.
    For N=2 this is trivial; for small N brute force is fine.
    """
    O = np.abs(v_ref.T @ v_t)  # (N,N)
    N = O.shape[0]
    best_p, best_score = None, -np.inf
    for p in itertools.permutations(range(N)):
        score = sum(O[i, p[i]] for i in range(N))
        if score > best_score:
            best_score, best_p = score, p
    return list(best_p)

def gevp(
    G, t0, ts=None, sort="Eigenvalue",
    vector_obs=True, symmetrise=True
):
    """
    Solve the GEVP for correlator matrices stored as:
      G[t][i][j] = Jackknife

    Parameters
    ----------
    G : list length T
        G[t] is NxN matrix (list-of-lists or np.array) of Jackknife
    t0 : int
        reference time for RHS: G(t) v = lam G(t0) v
    ts : int or None
        needed if sort is None or sort == "Eigenvector"
    sort : "Eigenvalue" | "Eigenvector" | None
    vector_obs : bool
        if True, return eigenvector components as Jackknife (uncertainties propagated)
        if False, return eigenvectors as float arrays (means only)
    symmetrise : bool
        if True, symmetrise G(t) sample-by-sample: (M + M^T)/2
    Returns
    -------
    lambdas : list of length N
        lambdas[s][t] is a Jackknife (or None)
    vecs : list of length N
        vecs[s][t] is either:
          - list of N Jackknife components (if vector_obs=True), or
          - numpy array shape (N,) of floats (if vector_obs=False),
        or None when unsolved.
    """

    T = len(G)
    G0_mat = np.asarray(G[t0], dtype=object)
    N = G0_mat.shape[0]

    JackknifeClass = type(G0_mat[0, 0])

    # --- helpers to build sample matrices at a given t ---
    def mat_samples_at_t(t):
        Mt = np.asarray(G[t], dtype=object)
        if Mt.shape != (N, N):
            raise ValueError(f"G[{t}] has shape {Mt.shape}, expected {(N, N)}")

        # number of jackknife samples: must match across i,j and also match t0
        ns = _get_jackknife_samples(Mt[0, 0]).shape[0]
        M = np.zeros((ns, N, N), dtype=float)

        for i in range(N):
            for j in range(N):
                s = _get_jackknife_samples(Mt[i, j])
                if s.shape[0] != ns:
                    raise ValueError(
                        f"Mismatch JK samples at t={t}, element ({i},{j}): {s.shape[0]} vs {ns}"
                    )
                M[:, i, j] = s

        if symmetrise:
            M = _symmetrise(M)
        return M, ns

    # Build G0 samples and remember ns0
    G0_s, ns0 = mat_samples_at_t(t0)

    def solve_time(t):
        Gt_s, ns = mat_samples_at_t(t)
        if ns != ns0:
            raise ValueError(
                f"Different number of JK samples between t0={t0} (ns={ns0}) and t={t} (ns={ns}). "
                "GEVP with JK propagation requires consistent resampling."
            )

        lam_s = np.zeros((ns0, N), dtype=float)
        v_s   = np.zeros((ns0, N, N), dtype=float)  # (sample, component, state)

        for k in range(ns0):
            w, v = _gevp_one_cholesky(Gt_s[k], G0_s[k])
            lam_s[k, :] = w
            v_s[k, :, :] = v

        return lam_s, v_s

    # --- output containers: per state, list over times ---
    lambdas = [[None] * T for _ in range(N)]
    vecs    = [[None] * T for _ in range(N)]

    def pack_time_into_outputs(t, lam_s, v_s):
        for s in range(N):
            lambdas[s][t] = JackknifeClass.from_samples(lam_s[:, s])
            if vector_obs:
                vecs[s][t] = [JackknifeClass.from_samples(v_s[:, c, s]) for c in range(N)]
            else:
                vecs[s][t] = np.mean(v_s[:, :, s], axis=0)


    # --- sort=None: solve only at ts ---
    if sort is None:
        if ts is None:
            raise ValueError("ts is required if sort=None.")
        lam_s, v_s = solve_time(ts)
        pack_time_into_outputs(ts, lam_s, v_s)
        return lambdas, vecs

    # --- solve all t > t0 with eigenvalue sorting (default) ---
    lam_all = [None] * T
    vec_all = [None] * T

    for t in range(t0 + 1, T):
        try:
            lam_s, v_s = solve_time(t)
            lam_all[t] = lam_s
            vec_all[t] = v_s
        except Exception:
            lam_all[t] = None
            vec_all[t] = None

    # --- optional eigenvector tracking against reference ts ---
    if sort == "Eigenvector":
        if ts is None:
            raise ValueError("ts is required for sort='Eigenvector'.")
        if not (t0 < ts < T):
            raise ValueError("ts must satisfy t0 < ts < T.")

        lam_ref_s, v_ref_s = solve_time(ts)

        for t in range(t0 + 1, T):
            if vec_all[t] is None:
                continue
            lam_t_s = lam_all[t]
            v_t_s   = vec_all[t]

            lam_new = np.zeros_like(lam_t_s)
            v_new   = np.zeros_like(v_t_s)

            for k in range(ns0):
                p = _best_permutation_by_overlap(v_ref_s[k], v_t_s[k])
                for s in range(N):
                    lam_new[k, s] = lam_t_s[k, p[s]]
                    v_new[k, :, s] = v_t_s[k, :, p[s]]

            lam_all[t] = lam_new
            vec_all[t] = v_new

    elif sort != "Eigenvalue":
        raise ValueError("Unknown sort. Use 'Eigenvalue', 'Eigenvector', or None.")
    
    # --- pack results into Jackknife outputs ---
    for t in range(t0 + 1, T):
        if lam_all[t] is None:
            continue
        pack_time_into_outputs(t, lam_all[t], vec_all[t])
        
    return lambdas, vecs


def project_state(G, vecs, state=0):
    """
    Full projection: C_state(t) = v^T G(t) v

    Parameters
    ----------
    G : list length T
        G[t] is NxN matrix of Jackknife objects
    vecs : output of gevp_yaac
        vecs[state][t] is [Jackknife components] if vector_obs=True
    state : int
        which state to project (0 = ground)

    Returns
    -------
    Cproj : list length T
        Cproj[t] is a Jackknife (or None)
    """    
    T = len(G)
    N = np.asarray(G[0], dtype=object).shape[0]
    Cproj = [None] * T

    for t in range(T):
        v = vecs[state][t]
        if v is None or G[t] is None:
            continue

        v_s = np.stack([vk.jk_samples for vk in v], axis=1)  # (ns, N)

        Gt = np.asarray(G[t], dtype=object)
        ns = v_s.shape[0]
        G_s = np.zeros((ns, N, N), dtype=float)
        for i in range(N):
            for j in range(N):
                G_s[:, i, j] = Gt[i, j].jk_samples
        C_s = np.einsum("ki,kij,kj->k", v_s, G_s, v_s)

        # Compute theta from full-sample eigenvector and correlator  ← new
        v_theta = np.array([vk.theta for vk in v])
        G_theta = np.array([[Gt[i, j].theta for j in range(N)] for i in range(N)])
        theta = v_theta @ G_theta @ v_theta  # ← new

        Cproj[t] = type(Gt[0, 0]).from_samples(C_s, theta=theta)  # ← theta added

    return Cproj


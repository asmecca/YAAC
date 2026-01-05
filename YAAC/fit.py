#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.style.use('./myplot.mplstyle')

class Jackknife:
    """
    Modern jackknife estimator for linear and non-linear observables.
    """

    def __init__(self, data, estimator= lambda x:np.mean(x)):
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

        self.N = self.data.shape[0]

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
        diff = self.jk_samples - self.mean
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
    def from_samples(cls, jk_samples):
        """
        Construct a Jackknife object directly from jackknife samples.
        """
        obj = cls.__new__(cls)

        obj.jk_samples = np.asarray(jk_samples)
        obj.N = obj.jk_samples.shape[0]

        obj.mean = np.mean(obj.jk_samples, axis=0)

        diff = obj.jk_samples - obj.mean
        obj.var = (obj.N - 1) / obj.N * np.sum(diff**2, axis=0)
        obj.std = np.sqrt(obj.var)

        # For derived objects, theta is the mean estimator
        obj.theta = obj.mean
        obj.unbiased = obj.mean

        # These are undefined / unused here
        obj.data = None
        obj.estimator = None

        return obj

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

    fit_params = []
    chi2_red_median = []

    # Fit window
    if fit_range is not None:
        tmin, tmax = fit_range
        x_fit = x[tmin:tmax]
    else:
        x_fit = x
        tmin, tmax = 0, len(x)

    for corr in corrs:
        data = corr.samples[:, tmin:tmax]   # (Njack, Nt_fit)
        Nj, Nt = data.shape
        npar = len(p0)

        params_jack = np.zeros((Nj, npar))
        chi2_red = np.zeros(Nj)

        # Covariance handling
        if correlated:
            if cov is None:
                raise ValueError("Covariance matrix must be provided for correlated fits.")
            cov_fit = cov[tmin:tmax, tmin:tmax]
            #cov_fit += 1e-12 * np.eye(Nt)
            L = np.linalg.cholesky(cov_fit)

        dof = Nt - npar

        for i in range(Nj):
            y = data[i]

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

        fit_params.append(params_jack)
        chi2_red_median.append(np.median(chi2_red))

    return fit_params, chi2_red_median
    

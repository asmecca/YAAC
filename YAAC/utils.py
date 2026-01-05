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

def _get_jackknife_samples(jk):
    return np.asarray(jk.jk_samples)

    
def read_corr(filename):
    # Reading Correlator
    corr=[]
    t=[]
    f=open(filename,'r')

    with open(filename) as g:
        first_line = g.readline().strip('\n')
        x=first_line.split()
        cnfg=int(x[0])
        time=int(x[1])
        T=int(x[2])
    g.close()
    
    for line in f.readlines():
        x=line.split()
        if x[0] != str(cnfg):
            corr += [float(x[0])]
    f.close()

    C = np.zeros((cnfg,time),dtype=float)
    for i in range(0,cnfg):
        for t in range(0,time):
            C[i][t] = corr[i*time + t]
    t_C = C.T
    jk_corr = [None]*time
    for t in range(0,time):
        jk_corr[t] = Jackknife(t_C[t])
    return jk_corr

def plot_corr(corr,xlabel,ylabel,yscale=None,data_label=None,color='blue',marker='o',ncol=1,save=None):
    for t in range(0,len(corr)):
        if corr[t] is not None:
            if t==0 and data_label is not None:
                plt.errorbar(x=t,y=corr[t].mean,yerr=corr[t].std,color=color,fmt=marker,label=data_label)
            else:
                plt.errorbar(x=t,y=corr[t].mean,yerr=corr[t].std,color=color,fmt=marker)
    if data_label is not None:
        plt.legend(loc='best',ncol=ncol)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if yscale is not None:
        plt.yscale(yscale)
    if save is not None:
        fig=plt.gcf()
        fig.savefig(save)
    plt.show()

def plot_multi_corr(list_corr,xlabel,ylabel,ylim=None,yscale=None,list_label=None,ncol=1,save=None,x_offset=None):
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if ylim is not None:
        y_i, y_f = ylim
        plt.ylim(y_i,y_f)
    if yscale is not None:
        plt.yscale(yscale)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 's', '^', 'v', 'D', '*', 'P', 'X']
    for i in range(0,len(list_corr)):
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
        for t in range(0,len(corr)):
            if t==0 and data_label is not None:
                plt.errorbar(x=t+i*offset,y=corr[t].mean,yerr=corr[t].std,color=color,fmt=marker,label=data_label)
            else:
                plt.errorbar(x=t+i*offset,y=corr[t].mean,yerr=corr[t].std,color=color,fmt=marker)
        if data_label is not None:
            plt.legend(loc='best',ncol=ncol)
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

        diff = samples - mean
        return (N - 1) / N * np.sum(diff * diff, axis=0)

    # --------------------------------------------------
    # Case 2: array / list of Jackknife objects
    # --------------------------------------------------
    jk_list = list(jk)
    Nt = len(jk_list)

    # Consistency check (important!)
    N = jk_list[0].N
    for j in jk_list:
        if j.N != N:
            raise ValueError("All Jackknife objects must have the same N")

    # Stack jackknife samples: shape (Njack, Nt, ...)
    samples = np.stack([j.jk_samples for j in jk_list], axis=1)

    # Mean over jackknife samples
    mean = np.mean(samples, axis=0)

    # Reshape to (Njack, Nt) for correlator-like objects
    diff = samples - mean

    # Covariance: (Nt x Nt)
    cov = (N - 1) / N * np.tensordot(diff, diff, axes=(0, 0))

    return cov

def jack_add(jk1,jk2):
    """
    Add two Jackknife objects sample-by-sample.

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

    samples = jk1.jk_samples + jk2.jk_samples
    return Jackknife.from_samples(samples)

def add_corrs(corr1,corr2):
    if len(corr1) != len(corr2):
        raise ValueError("Correlators must have the same length")

    res=[None]*len(corr1)
    for t in range(0,len(corr1)):
        res[t] = jack_add(corr1[t],corr2[t])
    
    return res

def jack_mul(jk1,jk2):
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
    return Jackknife.from_samples(samples)

def jack_mul_d(jk1,d):
    """
    Multiply a Jackknife object and a double sample-by-sample.

    Parameters
    ----------
    jk1 : Jackknife
        Objects with identical N and compatible shapes

    Returns
    -------
    Jackknife
        New jackknifed object representing the product
    """
    if isinstance(d,float) is False:
        raise ValueError("d is not a float")

    samples = jk1.jk_samples * d
    return Jackknife.from_samples(samples)

def multiply_corrs(corr1,corr2):
    if len(corr1) != len(corr2):
        raise ValueError("Correlators must have the same length")

    res=[None]*len(corr1)
    for t in range(0,len(corr1)):
        res[t] = jack_mul(corr1[t],corr2[t])
    
    return res

def multiply_corr_d(corr1,d):
    if isinstance(d,float) is False:
        raise ValueError("d is not a float")

    res=[None]*len(corr1)
    for t in range(0,len(corr1)):
        res[t] = jack_mul_d(corr1[t],d)
    
    return res

def jack_div(jk1,jk2,check_zero=True):
    """
    Divide two Jackknife objects sample-by-sample.

    Parameters
    ----------
    jk1, jk2 : Jackknife
        Objects with identical N and compatible shapes

    Returns
    -------
    Jackknife
        New jackknifed object representing the division
    """

    if jk1.N != jk2.N:
        raise ValueError("Jackknife objects must have the same N")

    denom = jk2.jk_samples

    if check_zero:
        if np.any(denom == 0.0):
            raise ZeroDivisionError("Division by zero in jackknife samples")    

    samples = jk1.jk_samples / denom
    return Jackknife.from_samples(samples)

def divide_corrs(corr1,corr2):
    if len(corr1) != len(corr2):
        raise ValueError("Correlators must have the same length")

    res=[None]*len(corr1)
    for t in range(0,len(corr1)):
        res[t] = jack_div(corr1[t],corr2[t])
    
    return res



def find_root_newton(d, root_function, guess, tol=1e-15, maxiter=100):
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
        # Ratio C(t) / C(t+1)
        jk_ratio = jack_div(jack_C[t], jack_C[t + 1])

        if method == "log":
            # Apply log replica-by-replica
            samples = np.log(jk_ratio.jk_samples)

        elif method == "cosh":
            guess = log_meff_guess(jack_C[t], jack_C[t + 1])
            if guess is not None:
                guess = abs(guess)

            if guess is None or not np.isfinite(guess):
                jack_meff.append(None)
                continue
            
            samples = []

            for R in jk_ratio.jk_samples:
                if R <= 0:
                    samples.append(np.nan)
                else:
                    samples.append(
                        meff_cosh_from_ratio(R, t, Nt, guess)
                    )

            samples = np.array(samples)
        else:
            raise NotImplementedError(
                f"Effective mass method '{method}' not implemented"
            )        

        # Build Jackknife object from derived samples
        jk_meff = Jackknife.from_samples(samples)
        jack_meff.append(jk_meff)

    return jack_meff


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

    #fit_params.append(params_jack)
    #chi2_red_median.append(np.median(chi2_red))

    return params_jack, np.median(chi2_red)
    

def format_with_error(value, error, nsig=2):
    """
    Format a value with uncertainty as x.xxx(yy).
    
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
    
    # Rounded error with nsig significant digits
    err_rounded = round(error, -exp + (nsig - 1))
    
    # Number of decimal places to show
    decimals = max(0, -(exp - (nsig - 1)))
    
    # Rounded value
    val_rounded = round(value, decimals)

    # Error in integer form
    err_int = int(round(err_rounded * 10**decimals))

    fmt = f"{{:.{decimals}f}}({{}})"
    return fmt.format(val_rounded, err_int)


#######
# TODO:
# - Fit correlated and uncorrelated
#######

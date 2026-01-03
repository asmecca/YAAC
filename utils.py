#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

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
    return C,jk_corr

def plot_corr(corr,xlabel,ylabel,yscale=None,data_label=None,color='blue',marker='o',ncol=1,save=None):
    for t in range(0,len(corr)):
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

#!/usr/bin/env python3
import numpy as np

class Jackknife:
    """
    Modern jackknife estimator for linear and non-linear observables.
    """

    def __init__(self, data, estimator):
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
    f=open(file_corr,'r')

    with open(file_corr) as g:
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
    return C

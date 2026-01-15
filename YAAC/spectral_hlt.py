#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import struct
import sys
import os

GEVFM=0.1973269804
NFUNC = 6
nsteps=1

dt_jack=np.dtype([('f','f8'),('n','i'),('m','f8'),('um','f8'),('s','f8'),('b','f8'),('boot','i')])

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


def get_rho(in_file):
    file_name=os.path.split(in_file)
    filename=file_name[0]+'/Rsigma_vec.bin'
    intsize = np.int32().nbytes
    doublesize = np.double().nbytes

    omega=[]
    eps,spec_sys=[],[]    
    spectre_sys=[]
    lamb,Bnorm,A0ABCW,A0ABCW_ref=[],[],[],[]
    jack_f=[]
    
    with open(filename, mode='rb') as file:        
        data = file.read(intsize+doublesize+intsize)
        neps,aM,nomega = struct.unpack('<idi', data)
        print('neps= ',neps)
        nomega=nomega-1
        print('nomega= ',nomega)
        for no in range(0,nomega):
            data = file.read(doublesize)
            tmp=struct.unpack('<d',data)[0]
            omega.append(tmp)
            for ieps in range(0,neps):
                data = file.read(doublesize)
                tmp=struct.unpack('<d', data)[0]
                if (no==0):
                    eps.append(tmp)
                #jack_store
                data=file.read(2*intsize)
                jack_boot,jack_n=struct.unpack('<ii',data)
                tmp = np.fromfile(file, count=(jack_n+4), dtype=np.double)
                jack_f.append(tmp)
    print('omega= ',omega)
    print('eps= ',eps)
    return jack_f,jack_boot,jack_n,neps,eps,nomega,omega
"""
    headersize = 2 * intsize + doublesize
    header = file.read(headersize)
    ns, s, nk = struct.unpack('>idi', header)
    print(f"{ns=}, {s=}, {nk=}")
    # nobs=input.ne*output.nk*input.ns
"""
def get_omega(in_file):
    file_name=os.path.split(in_file)
    filename=file_name[0]+'/Rsigma_omega0.300000.bin'
    intsize = np.int32().nbytes
    doublesize = np.double().nbytes

    omega=[]
    eps,spec_sys=[],[]    
    spectre_sys=[]
    lamb,Bnorm,A0ABCW,A0ABCW_ref=[],[],[],[]
    jack_f=[]
    
    with open(filename, mode='rb') as file:
        data=file.read(intsize+2*doublesize)
        neps,aM,omega=struct.unpack('<idd',data)
        print('neps from omega= ',neps)
        print('aM from omega= ',aM)        
        print('omega from omega= ',omega)
        for ieps in range(0,neps):
            data = file.read(doublesize)
            tmp=struct.unpack('<d', data)[0]
            eps.append(tmp)            
            data=file.read(3*intsize)
            nnorms,nsteps,nm=struct.unpack('<iii',data)
            print('nnorms= ',nnorms)
            print('nsteps= ',nsteps)
            print('nm= ',nm) 
            for i in range(0,nnorms):
                data_d = file.read(doublesize)
                tmp1=struct.unpack('<d',data_d)[0]
                spec_sys.append(tmp1)
                data_i = file.read(intsize)
                nmax = struct.unpack('<i',data_i)[0]
                print('nmax= ',nmax)
                for m in range(0,nmax):
                    data = file.read(2*doublesize)
                    tmp1,tmp2=struct.unpack('<dd',data)
                    lamb.append(tmp1)
                    Bnorm.append(tmp2)
                    tmp3=np.fromfile(file, count=NFUNC, dtype=np.double)
                    tmp4=np.fromfile(file, count=NFUNC, dtype=np.double)
                    A0ABCW.append(tmp3)
                    A0ABCW_ref.append(tmp4)
                    #jack_store
                    data=file.read(2*intsize)
                    jack_boot,jack_n=struct.unpack('<ii',data)
                    tmp = np.fromfile(file, count=(jack_n+4), dtype=np.double)
                    jack_f.append(tmp)
    return jack_f,jack_boot,jack_n,nnorms,neps,jack_n,eps

def read_rho_hlt(path,sig):
    rho_samples_sigma=[]
    rho_raw,rho_boot,nm,neps,eps_array,nomega,omega=get_rho(path)

    for iom in range(0,nomega):
        for ieps in range(0,neps):
            if float(eps_array[ieps])==float(sig):
                rho_samples_sigma+=[rho_raw[(iom*neps)+ieps]]

    yaac_rho = [None]*nomega
    for i in range(0,nomega):
        yaac_rho[i] = Jackknife.from_samples(rho_samples_sigma[i][:len(rho_samples_sigma[i])-4])
    return omega, yaac_rho

""" linear and nonlinear filters for density field reconstruction

Ben Granett ben.granett@brera.inaf.it
"""
import sys
import time
import numpy as N
import pylab
from scipy import sparse
import Solver
import fftutils

import cPickle as pickle

class LognormPoiss (Solver.Solver):
    """ """

    def __init__(self, n, w, Nbar, signal,mu,delta=None,likelihood=None):
        """
        inputs:
        n - observed number of galaxies
        w  - weight
        cf - correlation function object
        centers - coordinates of cell centers
        outputs:
        delta - overdensity
        """

        self.statsfile = "stats.txt"

        self.shape = n.shape

        n = n.flatten()
        w = w.flatten()
        
        # compute overdensity delta
        # to be used as first guess
        if delta==None:
            d = n*0.
        else:
            assert(N.all(1+delta>0))
            d = N.log(1+delta.flatten())

        ii = signal==0
        self.signalinv = 1./signal
        self.signalinv[ii] = 0
        
        self.n = n
        self.x0 = d
        self.Nbar = Nbar
        self.w = w
        self.b = Nbar*w

        self.mu = mu
        self.nbareff = N.mean(Nbar*w)

        print "nbar eff",self.nbareff
        print "nbar data",N.mean(self.n)


        self.likelihood=likelihood


    def writestats(self, data, x):
        """ """
        l = 0
        if not self.likelihood==None:
            l = self.likelihood(x.reshape(self.shape))
        out = file(self.statsfile,"a")
        for v in data:
            print >>out, v,
        print >>out, l,
        print >>out,""
        out.close()

    def go(self,tol=1e-2,diagfile='diagnos.pickle'):
        s,flag, diagnostics = self.nonlinear_secant(maxloops=200, maxinnerloops=20,
                               eps=tol, sigma0=1e-10, reset_freq=100, callback=self.writestats)
        #self.diagfile = diagfile
        #s,flag,diagnostics = self.nonlinear_nr(maxloops=1500, maxinnerloops=100,
        #                                       eps2=1e-10,eps=tol,reset_freq=5)
        
        s = N.reshape(s,self.shape)

        pickle.dump((self.shape,self.Nbar,flag,diagnostics),
                    file(diagfile,'w'))
        
        return s

    def gradf(self, s):
        """ solve for s=log(1+x) """

        if not N.all(N.isfinite(N.exp(s))):
            print s.max()
            raise

        d = self.Nbar*self.w*N.exp(s) - self.n
        #print "N.mean(s)",N.mean(s),self.mu
        r = d + self.Sinvx(s-self.mu)

        return r


    def Hessf(self, s, x):
        """ Compute the Hessian matrix dot x """
        return self.Sinvx(x) + self.Nbar*self.w*N.exp(s) * x

    def precond_inv(self,  x):
        """ Compute the inv precond dot with x """
        sig = 1./(self.signalinv + self.nbareff)
        y = self.fourier_Ax(sig,x)
        return y



        

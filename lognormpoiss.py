""" linear and nonlinear filters for density field reconstruction

Ben Granett ben.granett@brera.inaf.it
"""
import sys
import time
import numpy as N
from scipy import sparse
import Solver

class LognormPoiss (Solver.Solver):
    """ """

    def __init__(self, n, w, Nbar, cf, centers, delta=None, batch=10000):
        """
        inputs:
        n - observed number of galaxies
        w  - weight
        cf - correlation function object
        centers - coordinates of cell centers
        outputs:
        delta - overdensity
        """

        # compute overdensity delta
        # to be used as first guess
        if delta==None:
            d = n*1./(w*Nbar) - 1
            d[d<-0.999] = -0.999
        else:
            d=delta.copy()

        print d.min(),d.max(),d.mean()

        
        # signal cov matrix from the correlation function
        #Signal = cf.getMatrix(centers,centers)

        self.centers = centers
        self.cf = cf
        
        self.n = n
        self.x0 = N.log(1+d)
        self.Nbar = Nbar
        self.w = w
        self.b = Nbar*w

        self.mu = -cf(0)/2.

        self.batch=batch
        #self.go = self.nonlinear_ben

    def go(self):
        s = self.nonlinear_ben(maxloops=100, maxinnerloops=20,
                               eps=1e-5, sigma0=1e-5, reset_freq=50)
        print "exponentiating"
        return N.exp(s)-1

    def gradf(self, s):
        """ solve for s=log(1+x) """
        s[s>1e2] = 1e2
        if not N.all(N.isfinite(N.exp(s))):
            print s.max()
            raise

        print N.exp(s).mean()
        d = self.Nbar*self.w*N.exp(s) - self.n
        #d = -self.n  # neglect signal
        r = self.computeSb(d) + s - self.mu

        print "r.mean()",r.mean()
        sys.exit()
        
        return r


    def Hessf(self, s):
        """ Compute the Hessian matrix """
        D = sparse.spdiags(self.Nbar*self.w*s,0,len(s),len(s)).tocsr()
        one = sparse.eye(len(s),len(s))


        return one + self.computeSM(D)
    
    def computeSb(self, x):
        """ """
        out = []
        
        m = len(self.centers)
        for j in range(m/self.batch+1):
            a = j*self.batch
            b = min((j+1)*self.batch, m)

            key = a
            Signal = self.cf.getMatrix(self.centers,self.centers[a:b],key=key).transpose()

            r = Signal*x
            #print "j",j,a,b,self.centers[a:b].shape,Signal.shape,r.shape

            out.append(r)

        return N.concatenate(out)

    def computeSM(self, M):
        """ matrix matrix multiplication """

        out = []
        m = len(self.centers)
        for j in range(m/self.batch+1):
            a = j*self.batch
            b = min((j+1)*self.batch, m)

            key = a
            Signal = self.cf.getMatrix(self.centers,self.centers[a:b],key=key).transpose()

            r = Signal*M

            out.append(r)

        r = sparse.vstack(out)

        return r

import numpy as N
import pylab
import hmc
import fftutils
import pickle
import bins

class Data:

    def __init__(self, data, weight, signal, length=None, nbar=None):
        """ """
        self.shape = data.shape
        self.nparam = N.prod(self.shape)
        self.length = length
        self.volume = N.prod(length)

        self.fftnorm = N.prod(self.shape)**.5
        self.cellvol = N.prod(length)*1./self.nparam

        self.data = data
        self.w = weight
        self.signal = signal*1./self.cellvol  # renorm to power/cell
        self.nbar = nbar*self.cellvol         # renorm to power/cell

        print "nbar",self.nbar

        self.signal[self.signal==0] = 1
        assert(N.all(self.signal>0))

        self.mass = 1./self.signal + self.nbar

        self.fftcounter = 0

    def lnlike(self,x):
        """ """
        xk = self.gofft(x)
        xk = N.abs(xk)
        prior = -0.5*N.sum(xk*xk/self.signal)

        ngal = self.w*self.nbar*N.exp(x)
        ii = self.w>0
        like = N.sum(self.data[ii]*(N.log(self.w[ii]*self.nbar)+x[ii]) - ngal[ii])

        out = like + prior
        return out

    def grad_lnlike(self,x):
        """ """
        xk = self.gofft(x)
        xk = xk/self.signal
        prior = -self.gofftinv(xk)
        prior = prior.real


        like = self.data -  self.w*self.nbar*N.exp(x)
        like[self.w==0] = 0

        out = like + prior
        return out

    def _p_dot_invmass(self,p,order=1):
        """ compute momentum and inv mass dot product """

        pk = self.gofft(p)
        if order==2:
            pk = N.abs(pk)
            pk = pk*pk
        pm = pk/self.mass

        if order==1:
            pm = self.gofftinv(pm).real

        return pm

    def draw_momentum(self):
        """ """

        while True:
            # generate random amplitudes
            # make sure none are exactly 0
            amp = N.random.uniform(0,1,self.shape)
            if N.all(amp>0): break

        phase = N.random.uniform(0,2*N.pi,self.shape)

        x = N.sqrt(-2*N.log(amp))*N.exp(1j*phase)

        grid = N.sqrt(self.mass)*x

        p = self.gofftinv(grid).real

        return p

    def gofft(self,x):
        """ """
        self.fftcounter += 1
        return fftutils.gofft(x)*self.fftnorm

    def gofftinv(self,x):
        """ """
        self.fftcounter += 1
        return fftutils.gofftinv(x)/self.fftnorm




def normtest():  
    import simbox

    k = N.logspace(-4,10,100)
    pk = k**-2
    pk[0] = 0

    assert(N.all(N.isfinite(pk)))

    step = 1.
    shape = N.array((1024,))
    length = N.array(shape)*step

    S = simbox.SimBox(k, pk, shape, length, cachefile=None)

    x = []
    for i in range(100):
        xk = N.abs(fftutils.gofft(S.realize()))*32
        x.append(xk)

    x = N.array(x)

    pylab.plot(N.mean(x*x,axis=0))
    pylab.semilogy(S.pkgrid)
    pylab.show()



def derivtest():
    import simbox

    k = N.logspace(-4,10,100)
    pk = k**-2
    pk[0] = 0

    assert(N.all(N.isfinite(pk)))

    step = 1.
    shape = N.array((1024,))
    length = N.array(shape)*step

    S = simbox.SimBox(k, pk, shape, length,applywindow=False,cachefile=None)
    delta = S.realize()

    nbar = 0.01
    data = nbar*(1+delta)
    weight=N.ones(len(data))
    
    D = Data(data,weight,S.pkgrid, length=length, nbar=nbar)

    x0 = D.lnlike(delta)

    h = -1e-5
    delta2 = delta.copy()
    delta2[0]+=h

    x1 = D.lnlike(delta2)

    print "calc",D.grad_lnlike(delta)[0]

    print "should be",(x1-x0)/h
    exit()


def go(nbar=10, nloops=100):
    import simbox

    k = N.logspace(-4,10,100)
    pk = 2e-1*k**-1.5

    assert(N.all(N.isfinite(pk)))

    step = 1.
    shape = N.array((16,16,16))
    length = N.array(shape)*step

    S = simbox.SimBox(k, pk, shape, length, applywindow=False, cachefile=None,lognorm=True)



    nbargrid = nbar*N.prod(length)*1./N.prod(shape)

    for i in range(100):
        delta = N.exp(S.realize()) - 1
        data = nbargrid*(1+delta)
        print data.min()

        data_n = N.random.poisson(data)
        ks,pks = fftutils.powerspectrum(data_n/nbargrid - 1,length)
        a=ks.flatten()
        kmin = a[a.argsort()[1]]
        pkbins = 10**N.linspace(N.log10(kmin),N.log10(ks.max()),10)

        bpk = bins.binit(ks,pks,pkbins)
        x = (pkbins[1:]+pkbins[:-1])/2.
        pylab.plot(x,bpk,c='g',alpha=0.2)



    weight = N.ones(len(data))
    #weight[50:100] = 0
    #data_n *= weight

    assert(N.all(data>0))


    #pylab.plot(data_n,c='k')
    #pylab.plot(data,c='r')

    #pylab.plot(S.pkgrid)
    #pylab.figure()

    D = Data(data_n,weight,S.pkgrid, length=length, nbar=nbar)

    guess = delta

    H = hmc.hmc(x0=guess,dataobj=D)
    #H.traceh=True
    H.nsteps_range=(100,101)
    H.eps_range=(-2.,-1.8)

    H.go(nloops=nloops)

    print "> number of FFT",D.fftcounter,D.fftcounter*1./nloops

    samples = H.samples
    samples = N.array(samples)
    meansamples = nbargrid*N.exp(N.mean(samples,axis=0))


    for s in samples:
        ks,pks = fftutils.powerspectrum(s,length)
        a=ks.flatten()
        kmin = a[a.argsort()[1]]
        pkbins = 10**N.linspace(N.log10(kmin),N.log10(ks.max()),10)

        bpk = bins.binit(ks,pks,pkbins)
        x = (pkbins[1:]+pkbins[:-1])/2.
        pylab.plot(x,bpk,c='dodgerblue',alpha=0.2)
    pylab.axhline(1./nbar)

    bpk = bins.binit(ks,S.pkgrid.flatten(),pkbins)
    pylab.plot(x,bpk,c='k',lw=2)

    ii = (k>=x[0])&(k<=x[-1])
    pylab.loglog(k[ii],pk[ii],c='k')


    pylab.figure()
    #pylab.plot(data_n,c='k')
    #pylab.plot(data,c='r')

    a = data.min()
    b = data.max()
    pylab.subplot(221)
    pylab.imshow(data[0],interpolation='nearest',vmin=a,vmax=b)
    pylab.colorbar()

    pylab.subplot(222)
    pylab.imshow(data_n[0],interpolation='nearest',vmin=a,vmax=b)
    pylab.colorbar()


    pylab.subplot(223)
    pylab.imshow(meansamples[0],interpolation='nearest',vmin=a,vmax=b)
    pylab.colorbar()

    pylab.subplot(224)
    resid = data[0] - meansamples[0]
    print "resid",N.std(resid)
    sig = N.sqrt(nbar)
    pylab.imshow(resid,interpolation='nearest')
    pylab.colorbar()

    #pylab.figure()
    #pylab.plot(data,c='orange')
    #pylab.plot(data_n,c='k')

    pylab.figure()
    pylab.plot(samples[:,0,0,0])
    pylab.show()

#derivtest()
#normtest()
go(nloops=1000)

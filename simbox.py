
import sys,os
import numpy as N
import pylab
import cPickle as pickle
import time
from scipy.interpolate import interp1d,UnivariateSpline,interp2d


import fftutils
import bins

class SimBox:
    """ """
    cfmax = 0.5
    bias = 1.2
    sigv = 2.
    f = 0.5

    def __init__(self, k=None, pk=None, shape=None, length=None, lognorm=False, applywindow=True, pkgrid=None, cachefile="cache/pkgrid_%s.pickle",
                 verbose=True):
        """ """
        self.verbose = verbose
        self.shape = shape
        self.length = length
        self.dim = len(shape)
        self.volume = N.prod(self.length)
        self.n = N.prod(shape)
        self.maxlength = N.sqrt(N.sum([a**2 for a in self.length]))
        self.step = self.length[0]*1./self.shape[0]
        self.cachefile = cachefile
        self.pkgrid = None
        self.k = k
        self.pk = pk

        self.lognorm = lognorm
        self.applywindow = applywindow

        print self.dim,self.length,self.maxlength

        self.cfmaxr = self.cfmax*length[0]

        self.pkgrid = pkgrid
        if pkgrid==None:
            self.makePkGrid()



    def makePkGrid(self):
        """Evaluate the power spectrum on the grid
        """
        if not self.cachefile==None:
            tag = "%i_%i_%i_%i"%(self.shape[0],self.shape[1],self.shape[2],self.length[-1])
            cachefile = self.cachefile%tag
            if os.path.exists(cachefile):
                print "loading from cache",cachefile
                self.pkgrid = pickle.load(file(cachefile))
                return

        if self.dim==3:
            kgrid = fftutils.kgrid3d(self.shape, self.length) # components of k in physical units
            j = N.sqrt(kgrid[0]**2+kgrid[1]**2+kgrid[2]**2)
            ii = j>0
            mu = N.zeros(j.shape)
            mu[ii] = kgrid[2][ii]/j[ii]
        elif self.dim==2:
            kgrid = fftutils.kgrid2d(self.shape, self.length)
            j = N.sqrt(kgrid[0]**2+kgrid[1]**2)
            ii = j>0
            mu = j*0.
            mu = kgrid[1][ii]/j[ii]
        else:
            kgrid = N.abs(fftutils.kgrid1d(self.shape, self.length))
            mu = N.zeros(len(kgrid))
            j = kgrid

        j.flat[0] = j.flat[1]
        mu.flat[0] = 0

        mu = mu.flatten()
        j = j.flatten()
    
        lj = N.log10(j)
        lj = lj.flatten()

        if self.verbose:
            print "min,max log10(k):", j.min(),j.max(),lj.min(),lj.max()

        p = lj*0.
        lk = N.log10(self.k)
    
        if self.verbose:
            print "min,max log10(k):",lk.min(),lk.max()

        #assert(lk.min()<lj.min())
        #assert(lk.max()>lj.max())
    
        p = 10**N.interp(lj,lk,N.log10(self.pk),left=0,right=0)

        # apply redshift space distortions incl. velocity dispersion
        # print "Rsd"
        #p *= (self.bias+self.f*mu**2)*N.exp(-(j*mu*self.sigv)**2)

        p[0] = 0
        p = p.reshape(self.shape)

        if self.applywindow:
            print "> applying cell window function"
            w = 1.
            for i in range(len(kgrid)):
                x = kgrid[i]*self.step/2.
                bad = x==0
                x[bad]= 1
                w = w*N.sin(x)/x
                w[bad] = 1
            #p[w**2<0.5] = 0
            p *= w**2

        if self.lognorm:
            "> Computing log power spectrum ~~~~~~~~~~"
            xi = 1./self.volume*fftutils.gofftinv(p.astype('complex')).real
            if not N.all(xi.real>-1):
                print >>sys.stderr, "!!!!"
                print >>sys.stderr, "!!!! simbox fatal error with log transform! P(k) amp is too high maybe..."
                exit(1)
            logxi = N.log(1+xi)
            print "xi0",logxi.flat[0]
            p = self.volume*N.abs(fftutils.gofft(logxi))
            



        if not self.cachefile==None:
            pickle.dump(p, file(cachefile,"w"))

        self.S0 = N.sum(p)/self.volume


        self.pkgrid = p
        self.kgrid = j


    def realize(self):
        """realize a random gaussian field"""
        if self.verbose:
            print "> dreaming of gaussian fields",self.shape
        t0 = time.time()

        while True:
            # generate random amplitudes
            # make sure none are exactly 0
            amp = N.random.uniform(0,1,self.n)
            if N.all(amp>0): break

        phase = N.random.uniform(0,2*N.pi,self.n)

        x = N.sqrt(-2*N.log(amp))*N.exp(1j*phase)
        x = x.reshape(self.shape)

        grid = N.sqrt(1./self.volume*self.pkgrid)*x
    
        grid.flat[0] = 0

        t1 = time.time()
        out = fftutils.gofftinv(grid)
        if self.verbose:
            print " % fft time:",time.time()-t1
            print "> done, seconds:",time.time()-t0
        return out.real



    def cf(self, plot=False):
        """ Compute the correlation function on the grid """

        xi = 1./self.volume*fftutils.gofftinv(self.pkgrid.astype('complex'))
        
        assert(N.allclose(xi.imag/xi.real,0, atol=1e-5))

        if self.dim==3:
            rr = fftutils.kgrid3d(self.shape, 
                                  2.*N.pi*N.array(self.shape)/N.array(self.length))
            x,y,z = rr
            r = N.sqrt(x**2+y**2+z**2)
        elif self.dim==2:
            rr = fftutils.kgrid2d(self.shape, 
                                  2.*N.pi*N.array(self.shape)/N.array(self.length))
            x,z = rr
            r = N.sqrt(x**2+z**2)
        elif self.dim==1:
            rr = fftutils.kgrid1d(self.shape, 
                                  2.*N.pi*N.array(self.shape)/N.array(self.length))
            r = N.abs(rr)
            z = 0

        mu = z/r
        mu.flat[0] = 0
        
        r = r.flatten()
        mu = N.abs(mu.flatten())
        xi = xi.flatten().real

        if False:
            print mu.min(),mu.max()
            print r.min(),r.max(),self.step

            bins = N.arange(0,r.max(),2*self.step)
            data = N.transpose([r*mu,r*N.sqrt(1-mu**2)])
            print data.shape,xi.shape
            assert(N.all(N.isfinite(xi)))
            print xi
            h,e = N.histogramdd(data,(bins,bins),weights=xi)
            c,e = N.histogramdd(data,(bins,bins))
            h = h*1./c
            pylab.imshow(N.log10(N.abs(h)),origin='lower',extent=(bins[0],bins[-1],bins[0],bins[-1]),interpolation='nearest')
            pylab.colorbar()
            pylab.show()

            #interper = interp2d(r, mu, xi)

            sys.exit()

        r = r.flatten()
        xi = xi.real.flatten()



        order = N.argsort(r)
        r = r[order]
        xi = xi[order]

        i = r.searchsorted(self.cfmaxr)

        print "** Interpolation bounds",r[:i].min(),r[:i].max()
        #interper = interp1d(r[:i],xi[:i],bounds_error=False,fill_value=0,kind='linear')

        assert(N.all(N.isfinite(r)))
        assert(N.all(N.isfinite(xi)))

        out = xi * 0
        r2 = out*0
        count = out*0
        j = 0
        out[0] = xi[0]
        r2[0] = r[0]
        count[0] = 1
        for i in xrange(1,len(r)):
            if N.abs(r[i] - r[i-1])>1e-10:
                j += 1

            out[j] += xi[i]
            r2[j] = r[i]
            count[j] += 1

        xi = out[:j]/count[:j]
        r = r2[:j]



        #xi *= N.exp(-r**2/2./sig**2*10)

        x = r[-10:]
        y = xi[-10:]

        fit = N.polyfit(N.log(x),y,1)
        xx = N.linspace(r[-1],self.maxlength,100)
        yy = N.polyval(fit,N.log(xx))

        xi = N.concatenate([xi,yy[1:]])
        r = N.concatenate([r,xx[1:]])

        sig = N.min(self.length)/2.

        #xi *= N.exp(-r**2/2./sig**2)

        #fit = N.polyfit(N.log(r[1:i]), xi[1:i],5)
        #interper = lambda x: N.polyval(fit, N.log(x))
        interper = UnivariateSpline(N.log(1e-3+r),xi,k=3,s=.001)
        #interper = interp1d(r,xi)

        tran = lambda x: interper(N.log(1e-3+x))*N.exp(-x**2/2./sig**2)
        #tran = interper


        if plot:
            print "fuck",r.min(),r.max()
            pylab.plot(r,N.abs(xi),".")
            print "maxlength",self.maxlength
            rr = N.arange(r.min(),self.maxlength,self.step/10.)
            pylab.loglog(rr,N.abs(tran(rr)),'k-')
            pylab.show()
            sys.exit()
        
        return tran



        


def test():
    k = 10**N.arange(-4,10,.1)
    pk = k**-3/9.
    shape = (100,100,1)
    length = (100.,100.,1.)

    B = SimBox(k,pk,shape,length,cachefile=None)


    x = B.realize()

    pylab.imshow(x[:,:,0])
    pylab.colorbar()

    pylab.figure()
    pylab.hist(x.flatten(),bins=100)
    print "std",N.std(x.flatten())
    print 1./x.std()

    ko,pko = fftutils.powerspectrum(x,length)
    print ko.min(),ko.max()
    print "pk",pko.mean(),pko.mean()**.5

    bands = bins.logbins(ko.min(),ko.max(),20, )
    bx = (bands[1:]+bands[:-1])/2.
    pkb = bins.binit(ko,pko,bands)

    pylab.figure()
    ii = N.logical_and(k>=ko.min(), k<=ko.max())
    pylab.loglog(k[ii],pk[ii])
    pylab.plot(bx,pkb,".")
    
    pylab.show()


def testcf():
    k = 10**N.arange(-5,10,.1)
    pk = k**-3/9.

    B  = SimBox(k,pk, (100,100,100), (100,100,100), cachefile=None)

   
    xi = B.cf()

    x = N.arange(0,100,.1)
    pylab.plot(x,xi(x))


    
    pylab.show()

if __name__=="__main__":
    testcf()

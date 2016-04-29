import sys,os
import fftw3
from fftw3 import wisdom
import numpy as N
import cPickle as pickle
import time

""" Wrapper for pyfftw3 wrapper. """

wisdompath = '/tmp/fftwisdom%i'%os.getpid()


class FFT:
    """ An fftw3 wrapper class. On initialization it creates the plan, then you can call it """

    awhile = 0.01 # A time limit for saving wisdom.  If a plan takes
                  # longer than this, it is likely that it is the
                  # first time we have encountered the problem and the
                  # new wisdom should be saved.

    
    def __init__(self, x, real=False, direction='forward', nthreads=12, verbose=False):
        """ Create a plan for the FFT and remember it. """
        self.verbose = verbose
        self.shape = x.shape
        self.dtype = x.dtype

        self.outshape = self.shape
        outtype = 'complex'
        realtypes = None
        if real:
            if  direction[0] == 'f':
                assert(x.dtype == N.zeros(1,dtype='d').dtype)
                realtypes = 'halfcomplex r2c'
                outtype = 'complex'
                self.outshape = N.array(self.shape)
                self.outshape[-1] = self.shape[-1]//2+1
            elif direction[0] == 'b':
                realtypes = 'halfcomplex c2r'
                outtype = 'd'
            else:
                print "> fftw: can't understand direction",direction
                sys.exit(1)
        else:
            #print x.dtype
            assert(x.dtype == N.zeros(1,dtype='complex').dtype)
        
        self.outarray = N.zeros(self.outshape, dtype=outtype)
        temp = N.zeros(self.shape, dtype=x.dtype)

        loadwisdom(verbose=self.verbose)

        if self.verbose: print "> FFTW: constructing plan..."
        t0 = time.time()
        self.plan = fftw3.Plan(temp, self.outarray,
                               direction=direction,
                               flags=['measure'],
                               nthreads=nthreads,
                               realtypes=realtypes)
        t = time.time() - t0
        if self.verbose: print "> FFTW: plan made in %g sec"%t

        if t > self.awhile:
            if self.verbose:
                print "> FFTW: will write this bit of wisdom because it took a while to come up with."
            dumpwisdom(verbose=self.verbose)

    def go(self, x):
        """ Actually compute the FFT.

        The input array should be the same shape as the array used to

        construct the plan.

        The output will overwrite previous output.
        """
        assert(N.all(x.shape == self.shape))
        assert(x.dtype == self.dtype)
        self.plan.guru_execute_dft(x, self.outarray)        
        return self.outarray

    # just call the class instance and it will execute the fft!
    __call__ = go


cache = {}

def fft(x, shape=None, direction='forward', nthreads=1, real=False):
    global cache

    #print "shape",shape
    #print "*** hey!  Calling fft with dtype",x.dtype

    if not shape==None:
        inp = N.zeros(shape, x.dtype)
        s = tuple([slice(a) for a in x.shape])
        inp[s] = x
    else:
        inp = x
    
    s = reduce(lambda a,b:str(a)+"_%i"%b, inp.shape)
    key = "%s%i_%s"%(direction[0],nthreads,s)

    #print "> FFTW cache",key,cache.keys()
    if cache.has_key(key):
        #print "> FFTW: cache hit the plan!,",key
        plan = cache[key]
    else:
        #print "> FFTW: creating plan"
        plan = FFT(inp, direction=direction, nthreads=nthreads, real=real)
        cache[key] = plan

    #print "executing fftw"
    return plan(inp)

def ifft(x, shape=None, nthreads=1, real=False):
    return 1./N.prod(x.shape)*fft(x, shape=shape, direction='backward', nthreads=nthreads, real=real)

def dumpwisdom(path=wisdompath, verbose=False):
    """ """
    if verbose: print "> FFTW: writing fft wisdom to file",path
    if os.path.exists(path): # It doesn't seem to overwrite the file,
        os.unlink(path)      # so delete it first to ensure 

    wisdom.export_wisdom_to_file(path)

def loadwisdom(path=wisdompath, verbose=False):
    """ """
    if verbose: print "> FFTW: reading fft wisdom from file",path
    if os.path.exists(path):
        wisdom.import_wisdom_from_file(path)



def realtest():
    x = N.arange(16, dtype='d')
    y = fft(x, real=True)

    y2 = N.fft.rfft(x)

    
    print y
    print y2

    assert(N.allclose(y,y2))


def test2(n=2**7, dim=3):
    """ """
    s = tuple([n]*dim)
    n = N.prod(s)

    inputa = N.zeros(s, dtype='complex')
    
    f = N.random.normal(0,1,s,)
    inputa += f

    t0 = time.time()
    y = fft(inputa, nthreads=4)
    t1 = time.time()

    x = ifft(y, nthreads=4)
    t2 = time.time()

    print t1-t0
    print t2-t1

    assert(N.allclose(inputa,x))
    print "> pass"

def test(loops=10, n=2**8, dim=3):
    """ Do a speed test of FFTW against numpy.fftpack """
    s = tuple([n]*dim)

    inputa = N.zeros(s, dtype='complex')

    f = N.random.normal(0,1,s)
    inputa += f

    t0 = time.time()
    ft = FFT(inputa, nthreads=1)

    for i in range(loops):
        ya = ft.go(inputa)
    t1 = time.time()

    t2 = time.time()
    for i in range(loops):
        yb = N.fft.fftn(inputa)

    t3 = time.time()

    a = (t1-t0)/loops
    b = (t3-t2)/loops

    print "size = %i**%i"%(n,dim)
    print "mean of %i loops"%loops
    print "  fftw  ", a
    print "fftpack ", b
    print "speed-up",b/a


    assert(N.allclose(ya, yb))
    print '> test passed'


if __name__=="__main__":
    #realtest()
    test2()
    test()
    #realtest()

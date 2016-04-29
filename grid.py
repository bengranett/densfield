""" Routines to manage a discrete grid
Ben Granett ben.granett@brera.inaf.it
"""


import sys,os
import numpy as N
import cPickle as pickle

import fftutils
from scipy import ndimage

import scipy
import scipy.weave
import scipy.weave.inline_tools
import scipy.weave.c_spec
from scipy.weave.converters import blitz as cblitz


def ConstructGrid(coords, weights=None, step=1, min=None, max=None,
                  dim=None, noweight=False, savefile=None):
    """ Return a grid object """

    if min == None:
        min = coords.min(axis=0)
    if max == None:
        max = coords.max(axis=0)

    step = N.array(step)

    assert(len(min)==len(max))
    assert(len(step)==1 or len(step)==len(min))

    G = grid(min,max,step)
    G.putongrid(coords, weights=weights)

    print >>sys.stderr, G

    if not savefile==None:
        G.save(savefile)

    return G


def main():
    """ Load a catalogue from a file and bin it onto a grid. """

    import argparse

    def floatconv(s):
        a = s.replace(",", " ")
        a = [float(v) for v in a.split()]
        if len(a)==1:
            return a[0]
        return a
    
    parser = argparse.ArgumentParser(description='Put points on a grid')

    parser.add_argument('coord',  metavar='coordinate_file',
                        help='Input file, each line should have 1 coordinate tuple with n-numbers plus an optional weight')

    parser.add_argument('--out',  metavar='grid_file', default=sys.stdout,type=argparse.FileType('w', 0),
                        help='output file where grid will be saved (default stdout)')

    parser.add_argument('--dim', metavar='n', type=int, help='Grid dimension (default will be best guess)')

    parser.add_argument('--noweight', action='store_true', help='Do not apply weights!')

    parser.add_argument('--min',  metavar='a',nargs='+',type=floatconv,
                        help='grid min')

    parser.add_argument('--max',  metavar='b',nargs='+',type=floatconv,
                        help='grid max')

    parser.add_argument('--step',  metavar='c',nargs='+',type=floatconv, required=True,
                        help='size of cell')

    args = parser.parse_args()


    inpath = args.coord
    outpath = args.out
    min = args.min
    max = args.max
    step = args.step

    noweight = args.noweight
    dim = args.dim


    data = N.loadtxt(inpath, unpack=True)

    n = len(data)
    if dim==None:  # try to guess the dimension of the data
        if not min==None:
            dim = len(min)
        elif n>3: # assume last number is the weight
            dim = n-1
        elif noweight: # if noweight was specified, assume a weight was given
            dim = n-1
        else: 
            dim = n


    coords = data[:dim]
    weights = None

    if n>dim:
        weights = data[dim]

    if noweight: weights = None


    ConstructGrid(coords.transpose(),weights,min=min,max=max,step=step,savefile=outpath)
    





class grid:
    """ Grid class """
    finegrid = None   # super sampling parameters
    dosuper = False
    dogauss = False

    def __init__(self, min, max=None, step=None, shape=None, dtype='d', periodic=True):
        """ Initialize the bounds of the grid and other properties """
        self.dim = len(min)
        self.min = N.array(min).flatten()
        if not max==None: self.max = N.array(max).flatten()
        self.step = N.array(step).flatten()

        self.periodic = periodic

        if shape==None:
            #print "shape",(self.max - self.min)*1./self.step
            self.shape = N.ceil((self.max - self.min)/self.step).astype('int')
            #print "shape now",self.shape
        else:
            self.shape = shape
        self.ntot = N.prod(self.shape)

        # adjust max to include a full bin
        self.max = self.shape*self.step + self.min

        self.length = self.shape*self.step
        

        #self.store = N.zeros(self.shape,dtype=dtype)

        # mult contains multiplicative factors to compute the linear index
        # from an n-dimensional tuple of indices, see usage below
        mult = N.zeros(self.dim)
        for i in range(self.dim):
            mult[i] = N.prod(self.shape[i+1:])
        self.mult = mult

        self.key = "%i"%self.ntot

    def __str__(self):
        return "grid: %s %s %s %s"%(str(self.shape), str(self.min), str(self.max), str(self.step))

    def save(self, path):
        """ Save the data to a file """
        dump = (self.dim,self.min,self.max,self.step,self.shape,self.ntot,self.mult,self.key,self.data_b,self.weights)
        if type(path)==type('ciao'):
            pickle.dump(dump, file(path,'w'))
        else:
            pickle.dump(dump, path)

        print >>sys.stderr, "> dumped grid to",path

    def load(self, path):
        """ Load saved data """
        (self.dim,self.min,self.max,self.step,self.shape,self.ntot,self.mult,self.key,self.data_b,self.weights) = pickle.load(file(path))
        print >>sys.stderr, "> loaded grid from file",path

    def index(self, ii):
        """ compute linear index from tuple of n-dimensional indices """
        s = ii.shape

        if self.periodic:
            # enforce periodic boundary conditions
            ii = ii%self.shape

        ii = N.fix(ii).astype(int)

        outofbounds = N.min(ii,axis=1)<0
        for i in range(s[1]):
            outofbounds = N.logical_or(outofbounds, ii[:,i] >= self.shape[i])

        b = N.dot(ii,self.mult)
        b = N.fix(b).astype(int)
        outofbounds = N.logical_or(outofbounds,b>=self.ntot)
        inbounds = N.logical_not(outofbounds)
        return b, inbounds

    def ndindex(self, ind, shape=None):
        """ compute the n-dimensional index tuple from linear index"""

        mult = self.mult
        dim = self.dim

        if not shape==None:
            dim = len(shape)
            mult = N.zeros(dim)
            for i in range(dim):
                mult[i] = N.prod(shape[i+1:])
            
        ii = ind.copy()
        out = []
        for i in range(self.dim):
            a = ii//mult[i]  #integer division
            ii -= a*mult[i]
            out.append(a)
            
        out = N.transpose(out)
        return out#*self.step+self.min

    def bin2coord(self, index, shape = None,step=None, min=None):
        """ convert the n-dimensional index tuple to a tuple of coordinates """
        if step==None: step = self.step
        if min==None: min = self.min
        if shape==None: shape = self.shape
        return self.ndindex(index, shape)*step+self.min
        
    def bin(self, xyz):
        """ convert x,y,z coordinate to a linear bin index"""
        b = xyz*1./self.step-self.min*1./self.step
        #b = N.fix(b).astype('int')
        return self.index(b)

    def gridmask(self, xyz, mask, threshold=0.5):
        """ """
        self.putongrid(xyz, scheme='nearest')
        ii = mask.flat[self.data_b]>threshold
        return ii

    def putongrid(self, c, weights=None, scheme='nearest', **params):
        """ Bin a list of coordinates onto the grid.  Weights can be specified. """
        #print "putongrid",N.sum(weights)
        scheme = scheme.lower()
        if scheme.startswith('n'):
            b, inbounds = self.bin(c)
            self.data_b = b[inbounds]

            self.weights = weights
            if not weights==None:
                self.weights = weights[inbounds]
        elif scheme.startswith('c'):
            self.putongrid_cic(c,weights)
        elif scheme.startswith('s'):
            self.putongrid_super(c, weights=weights, **params)
        elif scheme.startswith('g'):
            self.putongrid_gauss(c, weights=weights, **params)
        else:
            print >>sys.stderr, "Uknown grid assignment scheme!",scheme
            exit(-1)


    def putongrid_cic(self, c, weights=None):
        """ """
        if weights==None:
            weights = 1

        a = self.min*1./self.step
        x0 = c*1./self.step - a  

        #assert(N.all(x0>=0))

        fp = x0 - N.trunc(x0)

        fp = fp.T - 0.5

        sign = N.ones(fp.shape)
        sign[fp<0] = -1

        sign = sign.T

        fp = N.abs(fp)

        
        s = 0
        ind = N.zeros(self.dim)

        pointlist = []
        weightlist = []

        self.data_b = []
        self.weights = []

        for i in range(2**self.dim):
                y = N.ones(fp.shape[1])  # y is len dim

                for k in range(self.dim):
                    if ind[k]==0: 
                        y *= 1-fp[k]
                    else:
                        y *= fp[k]
                
                #print i,N.trunc(x0) + ind*sign
                b,inbounds = self.index(N.trunc(x0)+ind*sign)
                self.data_b = N.concatenate([self.data_b,b])
                self.weights = N.concatenate([self.weights,y*weights])
                s += y
                

                ind[0] += 1
                for j in range(self.dim-1):
                    if ind[j]>1:
                        ind[j]=0
                        ind[j+1]+=1


            
    def putongrid_super(self, c, fact=2, weights=None):
        """ Do supersampling aliasing correction """
        self.dosuper = True

        if self.finegrid==None:
            print "making fine grid",self.shape*fact, self.step/fact
            self.finegrid = grid(min=self.min, shape=self.shape*fact, step=self.step/fact)

        self.finegrid.putongrid(c, scheme='cic', weights=weights)

    def render_super(self):
        """ render the super sampled grid"""
        h = self.finegrid.render()
        return downsample_fft(h, self.shape)



    def putongrid_gauss(self, c, fact=8, weights=None, sharp=4):
        """ """
        #print "put on grid gauss"
        self.dogauss = True
        self.gaussfact = fact
        self.sharp = sharp
        if self.finegrid==None:
            #print "making fine grid",self.shape*fact, self.step/fact
            self.finegrid = grid(min=self.min, shape=self.shape*fact, step=self.step/fact)

        #print "doing cic assignment"
        self.finegrid.putongrid(c, scheme='cic', weights=weights)


    def render_gauss(self, ):
        """ """
	sharp = self.sharp
        h = self.finegrid.render()
        shape = h.shape

        #print "fft"
        dk = fftutils.gofft(h)

        #print "kgrid"
        if len(shape)==3:
            kgrid = fftutils.kgrid3d_c(shape,N.ones(len(shape))*2*N.pi)
            k = N.sqrt(kgrid[0]**2+kgrid[1]**2+kgrid[2]**2)

        elif len(shape)==2:
            kgrid = fftutils.kgrid2d(shape,N.ones(len(shape))*2*N.pi)
            k = N.sqrt(kgrid[0]**2+kgrid[1]**2)

        elif len(shape)==1:
            kgrid = fftutils.kgrid1d(shape,N.ones(len(shape))*2*N.pi)
            #print kgrid.shape
            k = kgrid[0]
            #print "kshape",k.shape
            
        else:
            print "Wtf"
            exit(1)


	#print "low pass params",sharp,self.gaussfact
        kern = N.ones(k.shape)

        i = 0
        for kk in kgrid:
            klim = shape[i]/2./self.gaussfact
            #print "klim",klim
            ii = kk!=0
            kern[ii] *= 1./(1+N.exp(-sharp*N.log(N.abs(klim/kk[ii])) ))
            i+=1

        kern = kern.reshape(dk.shape)

        dk = dk*kern

        #print "inv fft"
        hsmoo = fftutils.gofftinv(dk).real

        n = N.log2(self.gaussfact).astype('int')
        #print "ntimes",n
        out = downsampleez(hsmoo, n)
        assert(N.all(N.isfinite(out)))
        return out
    


    def render(self):
        """ Produce an d-dimensional array representing the grid """
        if self.dosuper:
            return self.render_super()
        if self.dogauss:
            return self.render_gauss()

        g = N.zeros(self.ntot)
        ind = self.data_b
        weights = self.weights

        
        if weights==None:
            for i in xrange(len(ind)): g[ind[i]] += 1
        else:
            for i in xrange(len(ind)): g[ind[i]] += weights[i]

        return g.reshape(self.shape)



    def uniform(self,n=1, subi=0, subdim=None):
        """ return a uniform sampling of points n per cell """

        a = []
        for i,s in enumerate(self.shape):
            a.append(slice(0,s*n,1))
        o = N.mgrid[a]

        t = []
        for ax in o:
            t.append(ax.flatten())
        o = N.transpose(t)

        u = (o+.5/n)*self.step/n+self.min
        return u

    def uniform_generator(self, n=1, bunchsize=100):
        """ """
        highgrid = grid(self.min, step=self.step*1./n, shape=self.shape*n)
            
        shape = highgrid.shape
        print shape,self.shape*n

        step = self.step/n
               
        dim = self.dim

        ntot = N.prod(shape)
        h = int(ntot/bunchsize)
        
        for i in xrange(h+1):
            a = i*bunchsize
            b = (i+1)*bunchsize
            if b>ntot: b=ntot
            #print a,b,ntot
            ii = N.arange(a,b)
            c = highgrid.bin2coord(ii)+step/2.
            yield c


def downsample_fft(x, shape):
    """ down sample x to target shape """

    ntot = N.sum(x)
    f = N.prod(x.shape)*1./N.prod(shape)

    dk = fftutils.gofft(x)

    kx,ky,kz = fftutils.kgrid3d_c(shape,N.ones(len(shape))*2*N.pi)
    k2 = kx**2+ky**2+kz**2

    print k2.min(),k2.max(),shape

    klim2 = 0

    for j in range(len(shape)):

        xs = x.shape[j]
        s = shape[j]
        
        a = s//2 
        b = xs-s//2 

        klim2 += a**2

        ii = N.arange(a).astype('int')
        jj = N.arange(b,xs).astype('int')
        print "downsample len",len(ii),len(jj)

        ii = N.concatenate([ii,jj])
        assert(len(ii)==s)

        print dk.dtype,ii.dtype

        dk = N.take(dk, ii,axis=j)
	
    klim2=(min(shape)//2)**2
    print "klim",klim2
    k2 = k2.reshape(dk.shape)
    dk[k2>klim2] = 0

    print "zerod",N.sum(k2>klim2)

    dk *= f
    y = fftutils.gofftinv(dk).real

    ntot2 = N.sum(y)

    print "ratio",ntot/ntot2

    y = y*ntot*1./ntot2

    return y
    

def downsample(x, ntimes):
    """ """
    s = x.shape

    dim = len(s)
    y = x.copy()

    #print "downsample start", y.shape

    #print s

    for loop in range(ntimes):
        s = y.shape
        #print "shape",s
        for i in range(dim):
            sel1 = N.arange(0, s[i]-1, 2)
            sel2 = N.arange(1, s[i], 2)
            y = y.take(sel1,axis=i) + y.take(sel2, axis=i)
            y *= 0.5

    #print "downsample end",y.shape
    return y
    
def downsampleez(x, ntimes):
    """ """
    tot = x.sum()
    s = x.shape

    dim = len(s)
    y = x.copy()

    #print "downsample start", y.shape

    #print s

    #print "shape",s
    for i in range(dim):
        sel = N.arange(0, s[i], 2**ntimes)
        y = y.take(sel,axis=i)
        
    ytot = N.sum(y)

    #print tot,ytot
    assert(ytot>0)
    y *= tot*1./ytot

    #print "downsample end",y.shape
    return y


def blatest():
    """ Test routine """
    import time,pylab
    G = grid((0,0,0),(1,1,1),(.01,0.01,0.01))
    x = N.random.normal(0.5,0.1,1000000)
    y = N.random.normal(0.5,0.1,1000000)
    z = N.random.normal(0.5,0.1,1000000)

    weights = N.ones(len(z))
    
    c = N.transpose([x,y,z])
    G.putongrid(c, weights=weights)
    t0 = time.time()
    g = G.render()
    print time.time()-t0


    c = N.transpose([x,y,z])
    G.putongrid(c, weights=None)
    t0 = time.time()
    g2 = G.render()
    print time.time()-t0


    assert(N.allclose(g,g2))
    
    pylab.imshow(g[50])
    pylab.show()
    print G

def test_grid():
    G = grid((.1,.2,.3),(2,4,5),(.01,0.01,.01))
    xyz = [(.5,.5,.5),
           (.4,.3,.6),
           (.7,.2,.5)]
    xyz = N.array(xyz)
    i,bla = G.bin(xyz)
    xyz2 = G.bin2coord(i)

    assert(N.allclose(xyz,xyz2))

def test_uniform(n=3):
    import pylab
    import time
    
    G = grid((.1,.2),(.5,.6),(.001,0.0023333))
    print G
    u = G.uniform_generator(n, bunchsize=10003)
    g = N.zeros(G.shape)

    t0 = time.time()
    for a in u:
        G.putongrid(a)
        g +=  G.render()
    print "generator time", time.time()-t0

    k = n**G.dim

    t0 = time.time()
    G.putongrid(G.uniform(n))
    f = G.render()
    print "uniform time", time.time()-t0

    pylab.subplot(121)
    pylab.imshow(f)
    pylab.colorbar()
    pylab.subplot(122)
    pylab.imshow(g)
    pylab.colorbar()
    pylab.show()
    
    assert(N.allclose(f,g))
    
    assert(N.allclose(g,k))
    assert(N.allclose(g.sum(),G.ntot*k))



if __name__=="__main__":
    #main()
    test_uniform()

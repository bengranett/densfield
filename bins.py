import numpy as N
import time

import scipy
import scipy.weave
import scipy.weave.inline_tools
import scipy.weave.c_spec
from scipy.weave.converters import blitz as cblitz



def binit(x, y, bands, count=False):
    """ """

    out = N.zeros(len(bands)-1,dtype='d')
    n = out.copy()
    sig = out.copy()
    for i in range(len(out)):
        ii = (x >= bands[i]) & (x < bands[i+1])
        sub = y[ii]
        if sub.size > 0:
            out[i] = N.mean(sub)
            sig[i] = N.std(sub)
            n[i] += sub.size

    if count:
        return out,sig,n
    return out



def gocbinit(x, y, low, step, nbins):
    """ """
    
    h = N.zeros(nbins,dtype='d')
    c = N.zeros(nbins,dtype=int)
    n = len(x)

    step = float(step)
    low = float(low)
    nbins = int(nbins)

    code = \
    """
    int i,j;


    for (i=0;i<n;i++){
        j = (int)floor((x(i)-low)/step);
        if ((j >= 0) && (j < nbins)){
           h(j) += (double)y(i);
           c(j) += 1;
        }
    }
    for (i=0;i<nbins;i++) {
        if (c(i)>0)
           h(i) = h(i)/c(i);
    }


    """
    scipy.weave.inline(code,['n','x','y','low','step','nbins','c','h'],
                 extra_compile_args =['-O2 -fopenmp'],
                 extra_link_args=['-lgomp'],
                 headers=['<omp.h>','<math.h>'],
                 type_converters = cblitz,
                 compiler='gcc')

    return h;


def cbinit(x,y,bands,log=False):
    """ """
    if log:
        ex = N.log(x)
        ebands = N.log(bands)
        low = ebands[0]
        step = ebands[1]-ebands[0]
        nbins = len(ebands)-1
        return gocbinit(ex,y,low,step,nbins)
    return gocbinit(x,y,bands[0],bands[1]-bands[0],len(bands)-1)
        



def test():
    x = N.random.uniform(0,10,1e7)
    y = N.random.normal(100,10,len(x))
    low = 0
    step = 0.5
    nbins = 20
    #bands = N.arange(nbins+1)*step + low
    bands = 10**N.linspace(-3,1,20)

    t0 = time.time()
    h = binit(x,y,bands)
    t1 = time.time()
    print t1-t0

    t0 = time.time()
    h2 = cbinit(x,y,bands,log=True)
    t1 = time.time()
    print t1-t0

    print h
    print h2

    assert(N.allclose(h,h2))


def binit_lnorm(x, y, bands):
    """ """

    norm = x*(x+1)

    out = N.zeros(len(bands)-1,dtype='d')
    for i in range(len(out)):
        ii = N.where(x >= bands[i])
        jj = N.where(x[ii] < bands[i+1])
        sub = y[ii][jj]
        n = norm[ii][jj]
        if sub.size > 0:
            out[i] = N.sum(sub*n)/N.sum(n)

    return out


def logbins(min,max, n, integer = False):
    """integer log bins"""
    l = N.exp(N.arange(N.log(min),N.log(max),N.log(max/min)/n))
    if not integer:
        return l
    
    d = {}
    for a in l:
        d[int(a)] = 1
    out = d.keys()
    out.sort()
    return N.array(out)



def thetabins(f="cache/cross.cor",n=20):

    theta = []
    for line in file(f):
        if line.startswith("#"): continue
        theta.append(float(line.split()[0]))
    theta = N.array(theta)
    theta.sort()

    i = N.arange(len(theta))
    bins = logbins(1,len(i),n,integer=True)

    x = binit(i,theta,bins)

    if x[0] != theta[0]:
        x = N.concatenate([[theta[0]],x])


    return x



if __name__=="__main__":
    test()

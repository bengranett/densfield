import sys
import numpy as N
import time

import fftw


import scipy
import scipy.weave
import scipy.weave.inline_tools
import scipy.weave.c_spec
from scipy.weave.converters import blitz as cblitz






nthreads = 1

def powerspectrum(grid, length, mask=None, zeropad=None, norm=1, getdelta=False,computek=True,
                  nthreads=1):
    """ Compute the power spectrum 
    Inputs:
      grid -- delta values 1, 2 or 3 dimensions
      length -- physical dimensions of box
    Outputs:
      k, pk
    """
    shape = grid.shape
    dim = len(shape)

    if not zeropad==None:
        bigbox = N.zeros(N.array(grid.shape)*zeropad)
        if dim==3: bigbox[:grid.shape[0],:grid.shape[1],:grid.shape[2]] = grid
        if dim==2: bigbox[:grid.shape[0],:grid.shape[1]] = grid

        bigmask = None
        if not mask==None:
            bigmask = N.zeros(N.array(grid.shape)*zeropad)
            bigmask[:grid.shape[0],:grid.shape[1],:grid.shape[2]] = mask
        
        return powerspectrum(bigbox,N.array(length)*zeropad, mask=bigmask, zeropad=None, getdelta=getdelta, norm=zeropad**3, nthreads=nthreads,
                             computek=computek)

    if N.shape(length)==():          # True if length is a number
        length = N.ones(dim)*length  # create a list

    assert(len(length)==dim)

    t0 = time.time()
    dk = gofft(grid, nthreads=nthreads)

    #print "norm",norm
    dk *= N.sqrt(N.prod(length)*norm)

    if not mask==None:
        print "no use of a mask is implemented!"
    #print "fft time",time.time()-t0

    pk = N.abs(dk**2)
    pk = pk.flatten()


    # save significant time if we dont need to recompute k
    if not computek:
        #print "skipping k comptuation"
        if getdelta:
            return pk, dk
        return pk
    if dim==3:
        kgrid = kgrid3d(shape, length)
    elif dim==2:
        kgrid = kgrid2d(shape, length)
    elif dim==1:
        kgrid = kgrid1d(shape, length)
    else:
        print >>sys.stderr, "fftutils: bad grid dimension:",dim
        raise

    #print kgrid[0].max()
    s = 0
    for i in range(dim):
        s += kgrid[i]**2
    k = s.flatten()**.5

    pk = pk[1:]
    k = k[1:]
    assert(N.all(k>0))

    #print "kmax",k.max(),shape,length

    # sorting is pretty slow
    #order = k.argsort()
    #k = k[order][1:]
    #pk = pk[order][1:]

    if getdelta:
        return k, pk, (kgrid, dk)

    return k, pk

    

def gofft_numpy(grid, nthreads=1):
    """ Forward FFT """
    n = N.prod(grid.shape)
    dk = 1./n*N.fft.fftn(grid)
    return dk

def gofftinv_numpy(grid, nthreads=1):
    """ inverse FFT """
    n = N.prod(grid.shape)
    d = n*N.fft.ifftn(grid)
    return d

def gofft_fftw(grid, nthreads=1):
    """ Forward FFT """
    #print "gofft_fftw"
    n = N.prod(grid.shape)
    
    #print "size",grid.shape,grid.nbytes*1./1024**3
    if grid.dtype=='complex':
        print "ok complex"
        g = grid
    else:
        g = grid.astype('complex')

    dk = 1./n*fftw.fft(g, grid.shape, nthreads=nthreads)
    return dk

def gofftinv_fftw(grid, nthreads=1):
    """ inverse FFT """
    #print "WARNING!     gofftinv_fftw has not been tested!!!!"
    n = N.prod(grid.shape)
    d = n*fftw.ifft(grid, grid.shape,nthreads=nthreads)
    return d

gofft = gofft_fftw
gofftinv = gofftinv_fftw


def cbinpk(pk, shape, length,axis,low,step,nbins):
    """ """
    n = int(N.prod(shape));
    y = pk.flatten()

    h = N.zeros(nbins,dtype='d')
    c = N.zeros(nbins,dtype=int)
    step = float(step)
    low = float(low)
    nbins = int(nbins)

    sx,sy,sz = shape
    sx = int(sx)
    sy = int(sy)
    sz = int(sz)
    lx,ly,lz = length

    normx = float(2*N.pi*1./lx)
    normy = float(2*N.pi*1./ly)
    normz = float(2*N.pi*1./lz)

    code = \
    """
    int j;
    long i;
    int kx,ky,kz;
    double kk,tkx,tky,tkz;

    int hsx,hsy,hsz;
    hsx = int((double)sx/2);
    hsy = int((double)sy/2);
    hsz = int((double)sz/2);


    printf("cbin n=%ld\\n",n);

    kx=ky=kz=0;

    for (i=0;i<n;i++){
       switch (axis) {

         case 0:
          if (kx>hsx)
            kk = (double)(kx-sx)*normx;
          else
            kk = (double)kx*normx;
          break;

         case 1:
          if (ky>hsy)
            kk = (double)(ky-sy)*normy;
          else
            kk = (double)ky*normy;
          break;

         case 2:
          if (kz>hsz)
            kk = (double)(kz-sz)*normz;
          else
            kk = (double)kz*normz;
          break;

         default:
          if (kx>hsx)
            tkx = (double)(kx-sx)*normx;
          else
            tkx = (double)kx*normx;
          if (ky>hsy)
            tky = (double)(ky-sy)*normy;
          else
            tky = (double)ky*normy;
          if (kz>hsz)
            tkz = (double)(kz-sz)*normz;
          else
            tkz = (double)kz*normz;
          kk = (double)sqrt(tkx*tkx + tky*tky + tkz*tkz);
      }

      //printf("%d %d %d %d %lf\\n",i,kx,ky,kz,kk);

      // make histogram

      j = (int)floor((kk-low)/step);
      if ((j >= 0) && (j < nbins)){
           h(j) += (double)y(i);
           c(j) += 1;
      }


      // advance k indices

       kz += 1;
       if (kz>sz-1){
          kz=0;
          ky+=1;
         }
       if (ky>sy-1){
         ky=0;
         kx+=1;
       }

      

    }


    for (i=0;i<nbins;i++) {
        if (c(i)>0)
           h(i) = h(i)/c(i);
    }

"""

    scipy.weave.inline(code,['y','sx','sy','sz','normx','normy','normz','axis',
                             'n','low','step','nbins','c','h'],
                 extra_compile_args =['-O2'],
                 extra_link_args=[],
                 headers=['<math.h>'],
                 type_converters = cblitz,
                 compiler='gcc')

    return h


def test():
    import bins

    for loop in range(100):
        shape = N.random.randint(10,100,3)
        #shape = [3,3,3]
        for axis in [0,1,2,3]:
            length = N.array(shape)*1.0
            pk = N.random.normal(0,1,shape).flatten()

            low = 0
            step = .05
            nbins = 10
            
            bpk = cbinpk(pk, shape, length,axis,low,step,nbins)


            kk=kgrid3d(shape,length)
            if axis in [0,1,2]:
                k = kk[axis]
            else:
                k = N.sqrt(kk[0]**2+kk[1]**2+kk[2]**2)

            #print k.flatten()
            #print kk[axis].flatten()

            #k = N.sum(kx**2+ky**2+kz**2)
            bpk2 = bins.binit(k.flatten(),pk,N.arange(nbins+1)*step + low)

            try:
                assert(N.allclose(bpk,bpk2))
            except:
                print "fail!",shape,axis
                print bpk
                print bpk2
                exit()
            print "pass",shape,axis
    exit()



def kgrid3d_c(shape, length):
    """ """
    n = int(N.prod(shape));

    sx,sy,sz = shape
    sx = int(sx)
    sy = int(sy)
    sz = int(sz)
    lx,ly,lz = length

    normx = float(2*N.pi*1./lx)
    normy = float(2*N.pi*1./ly)
    normz = float(2*N.pi*1./lz)

    kxgrid = N.zeros(n,dtype='d')
    kygrid = N.zeros(n,dtype='d')
    kzgrid = N.zeros(n,dtype='d')

    code = \
    """
    long i;
    int kx,ky,kz;
    double tkx,tky,tkz;

    int hsx,hsy,hsz;
    hsx = int((double)sx/2);
    hsy = int((double)sy/2);
    hsz = int((double)sz/2);


    kx=ky=kz=0;

    for (i=0;i<n;i++){

          if (kx>hsx)
            tkx = (double)(kx-sx)*normx;
          else
            tkx = (double)kx*normx;
          if (ky>hsy)
            tky = (double)(ky-sy)*normy;
          else
            tky = (double)ky*normy;
          if (kz>hsz)
            tkz = (double)(kz-sz)*normz;
          else
            tkz = (double)kz*normz;

       kxgrid(i) = tkx;
       kygrid(i) = tky;
       kzgrid(i) = tkz;

      // advance k indices

       kz += 1;
       if (kz>sz-1){
          kz=0;
          ky+=1;
         }
       if (ky>sy-1){
         ky=0;
         kx+=1;
       }

    }


"""

    scipy.weave.inline(code,['sx','sy','sz','normx','normy','normz',
                             'n','kxgrid','kygrid','kzgrid'],
                 extra_compile_args =['-O2'],
                 extra_link_args=[],
                 headers=['<math.h>'],
                 type_converters = cblitz,
                 compiler='gcc')

    return kxgrid,kygrid,kzgrid


def testkgrid3d():

    ta = 0.
    tb = 0.
    for loop in range(100):
        shape = N.random.randint(10,100,3)
        length = shape*1.

        t0 = time.time()
        kx,ky,kz = kgrid3d(shape,length)
        ta += time.time()-t0

        t0 = time.time()
        ckx,cky,ckz = kgrid3d_c(shape,length)
        tb += time.time()-t0

        kx = kx.flatten()
        ky = ky.flatten()
        kz = kz.flatten()

        assert(N.allclose(kx,ckx))
        assert(N.allclose(ky,cky))
        assert(N.allclose(kz,ckz))
        print "pass"
    print "times",ta,tb

kgridcache = {}
def kgrid3d(shape, length):
    """ Return the array of frequencies """
    key = '%s %s'%(shape[0],length[0])
    if kgridcache.has_key(key):
        print "hitting up kgrid cache"
        return kgridcache[key]

    a = N.fromfunction(lambda x,y,z:x, shape)
    a[N.where(a > shape[0]//2)] -= shape[0]
    b = N.fromfunction(lambda x,y,z:y, shape)
    b[N.where(b > shape[1]//2)] -= shape[1]
    c = N.fromfunction(lambda x,y,z:z, shape)
    c[N.where(c > shape[2]//2)] -= shape[2]

    norm = 2*N.pi
    a = a*norm*1./length[0]
    b = b*norm*1./length[1]
    c = c*norm*1./length[2]

    kgridcache[key] = (a,b,c)

    return a,b,c

def kgrid2d(shape, length):
    """ Return the array of frequencies """
    
    a = N.fromfunction(lambda x,y:x, shape)
    a[N.where(a > shape[0]//2)] -= shape[0]
    b = N.fromfunction(lambda x,y:y, shape)
    b[N.where(b > shape[1]//2)] -= shape[1]

    norm = 2*N.pi
    a = a*norm*1./(length[0])
    b = b*norm*1./(length[1])

    return a,b

def kgrid1d(shape,length):
    """ Return the array of frequencies """
    a = N.arange(shape[0])
    a[N.where(a > shape[0]//2)] -= shape[0]
    a = a*2*N.pi*1./(length[0])
    return N.array([a])


def testpoisson(mu=9, shape=(30,30,30)):
    print "-------- Poisson test mu=%g, shape=%s ---------"%(mu,str(shape))
    grid = N.random.poisson(mu,shape)*1./mu - 1

    # forward and backward transform
    gk = gofft(grid)
    g2 = gofftinv(gk)
    assert(N.allclose(grid,g2))

    # test power spectrum
    k,pk = powerspectrum(grid, shape)
    
    print "  pk mean", pk.mean(), "should be", 1./mu
    print "  kmin, kmax", k.min(), k.max()
    print "  error",pk.mean()*mu-1.
    assert(N.abs(pk.mean()*mu-1.) < .05)
    print " pass :)"


def testmask(mu=100, shape=(30,30,30)):
    import pylab,bins
    grid = N.random.poisson(mu,shape)*1./mu - 1

    mask = N.fromfunction(lambda x,y,z: x+y+z<45, shape)*1.
    
    f = N.sum(mask)*1./mask.size

    k,p1 = powerspectrum(grid, shape, zeropad=2)
    k,p2 = powerspectrum(grid*mask, shape, zeropad=2)
    k,p3 = powerspectrum(grid, shape, zeropad=None)

    print "f",f
    p2 /= f

    print k
    kbins = bins.logbins(k.min(),k.max(),100)
    print kbins
    x = kbins[1:]
    p1b = bins.binit(k,p1,kbins)
    p2b = bins.binit(k,p2,kbins)
    p3b = bins.binit(k,p3,kbins)

    pylab.plot(x,p1b)
    pylab.plot(x,p2b)
    pylab.semilogy(x,p3b)

    pylab.axhline(1./mu)
    pylab.show()



if __name__=="__main__":
    testkgrid3d();
    sys.exit()
    x = N.random.normal(0,1,1e4)
    kx = gofft(x)
    x2 = gofftinv(kx)
    print x2
    assert(N.allclose(x,x2))
    print "pass"

    #testmask()
    #sys.exit()
    testpoisson(shape=(30,30,50))
    #testpoisson(shape=(200,200))
    #testpoisson(shape=4e4)


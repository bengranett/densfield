import sys
sys.path.append("/home/ben/bin")
sys.path.append("/home/ben/pylib/cosmopy")

import os.path

import numpy as N
import pickle
import time

import sphere

import theory
import theory.pt as Pt
import theory.frw as Frw
import theory.param as Param
import theory.utils as Utils
import theory.proj as Proj


class distance:

    cambpath = '/home/wp3a/camb'

    # HOD mock parameters


    def __init__(self, angles=None, **params):
        """ """
        self.h = 0.7
        p = {'omega_baryon': 0.045,
             'omega_cdm': 0.30-0.045,
             'omega_lambda': 0.70,
             'hubble': 100*self.h,
             'scalar_spectral_index': [0.95],
             }

        self.params = Param.CosmoParams(**p)

        self.angles = angles
        for k in params:
            print "set",k,params[k]
            self.params[k] = params[k]

        self.D = Frw.Distance(self.params)
        self.PS = Pt.PowerSpectrum(self.params)


    def radec2xyz(self,ra,dec,z, raddist=False, rotate=True,angles=None):
        """ Returns Mpc/h """

        if not raddist:
            r = self.D.rc(z)*self.h
        else:
            r = z

        x,y,z = sphere.lonlat2xyz(ra,dec,r)

        if rotate:
            if angles==None:
                angles = self.angles
                
            if angles==None:
                rac = (ra.min()+ra.max())/2.
                decc = (dec.min()+dec.max())/2.
                angles = [(-rac,90-decc),(270,0)]
            x,y,z = sphere.rotate_xyz(x,y,z,angles=angles)
            self.angles = angles
    
        return x,y,z

    def xyz2radec(self, x,y,z, rotate=True):
        """ """
        if rotate:
            x,y,z = sphere.rotate_xyz(x,y,z,angles=self.angles,inverse=True)

        ra,dec,r = sphere.xyz2lonlat(x,y,z)
        return ra,dec,r

    def rc(self, redshift):
        return self.D.rc(redshift)*self.h

    def dm(self, redshift):
        return self.D.dm(redshift) + 5*N.log10(self.h)

    def redshift(self, rc):
        return self.D.inv_rc(rc/self.h)

    def vc(self, redshift):
        return self.D.vc(redshift)*self.h**3

    def growth(self, redshift):
        return self.PS.d1(redshift)

    def H(self, redshift):
        """ return hubble constant at redshift """
        return 100*self.D.E(redshift)

    def omm(self, redshift):
        """ return omm at redshift """
        return (100./self.H(redshift))**2*(self.params['omega_cdm']+self.params['omega_baryon'])*(1+redshift)**3


def test_distance():
    D = distance()

    zz = N.linspace(0.,1,10)
    x = (1+zz)/D.H(zz)

    for i in range(len(zz)):
        print i,zz[i],x[i]


    sys.exit()
    
    ra = N.arange(0,180,10.)
    dec = N.random.uniform(50,60,len(ra))
    z = N.linspace(.1,1,len(ra))

    xyz = D.radec2xyz(ra,dec,z)
    a,b,r = D.xyz2radec(*xyz)

    print ra
    print a
    print dec
    print b
    assert(N.allclose(ra,a))
    assert(N.allclose(dec,b))

if __name__=="__main__":
    test_distance()

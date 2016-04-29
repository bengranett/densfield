import sys,os
import time
import cPickle as pickle

import numpy as N
import pylab

class Data:
    Cinv = N.linalg.inv(N.array([[1,0.5],[0.5,4.]]))
    nparam = 2
    mass = 1
    shape = (2,)

    def __init__(self):
        """ """

    def lnlike(self, q):
        """ compute likelihood """
        print "like",q
        l = -0.5*N.dot(q,N.dot(self.Cinv,q))
        return l

    def grad_lnlike(self, q):
        """ compute likelihood """
        print "grad",q
        g = -N.dot(self.Cinv,q)
        return g

    def hess_lnlike(self,q):
        """ """
        g = -N.sum(self.Cinv,axis=0)
        return g

    def _p_dot_invmass(self,p,order=1):
        """ compute momentum and inv mass dot product """
        print "fft",p
        pm = p/self.mass
        if order==2:
            pm *= p
        return pm

    def draw_momentum(self):
        """ """
        return N.random.normal(0,1,self.nparam)


class hmc:
    eps_range = -2,0
    nsteps_range = 3,4
    verbose = True

    def __init__(self, x0=0, dataobj=None, savedir='save'):
        """ """        
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        self.savedir = savedir

        self.D = dataobj

        self.samples = []
        self.momsamples = []
        self.times = []
        self.stepparam = []
        self.ham = []
        self.dHam = []
        self.acceptflag = []

        self.accept_count = 0
        self.count = 0
        self.q = x0



        self.traceh = False
        self.start = 0

        print ">> grid shape:",self.D.shape
        print ">> number of parameters:",self.D.nparam

    def restart_from_step(self, i):
        """ """
        self.q = N.load("%s/sample_%i.npy"%(self.savedir,i))
        self.count = i+1
        self.start = i+1

    def write_output(self):
        """ """
        if not self.acceptflag[-1]: return
        #stats = (self.ham,self.stepparam,self.times,self.acceptflag)
        #samples = self.samples
        #momsamples = self.momsamples

        #pickle.dump(stats, file("%s/stats_%i.pickle"%self.count,'w'))
        N.save("%s/sample_%i.npy"%(self.savedir,self.count), self.q)
        #N.save("%s/momentum_%i.npy"%(self.savedir,self.count), momsamples[-1])


    def write_log_txt(self):
        """ """
        H = 1./N.max(self.hessian(self.q))**.5

        out = file("%s/log.txt"%self.savedir,"a")
        print >>out, self.count, self.times[-1], self.ham[-1][0], self.ham[-1][1], self.dHam[-1], int(self.acceptflag[-1]), self.stepparam[-1][0],N.log10(self.stepparam[-1][1]),N.log10(H)
        out.close()

        out = file("%s/absdiff_%03i.txt"%(self.savedir,self.count),"w")
        for i in range(len(self.absdiff)):
            print >>out, self.absdiff[i]
        out.close()

    def pot(self, q):
        """ """
        U = - self.D.lnlike(q)
        return U

    def grad_pot(self, q):
        """ """
        gradU = -self.D.grad_lnlike(q)
        return gradU
        

    def kinetic(self, p):
        pk2 = self.D._p_dot_invmass(p,order=2)
        K = 0.5*N.sum(pk2)
        return K

    def hessian(self, q):
        """ """
        U = -self.D.hess_lnlike(q)
        print "Hess",N.mean(U)
        return U


    def seteps(self):
        """ """
        self.nsteps = N.random.randint(*self.nsteps_range)
        self.eps = 10**N.random.uniform(*self.eps_range)
        #determine eps range
        #self.eps = 1./N.mean(self.hessian(self.q))**.5/100
        #print "eps max",epsmax
        #self.eps = 10**(N.random.uniform(epsmax-0.5,epsmax))
        #print "step from hessian:",self.eps


    def step(self):
        """ """
        nsteps = self.nsteps
        eps = self.eps


        t0 = time.time()

        
        p = self.D.draw_momentum()

        p = p*(1-self.mask)

        q = self.q
        H_current = self.pot(q), self.kinetic(p)


        # half step
        p = p-eps/2.*self.grad_pot(q)

        H = []
        absdiff = []
        sum = N.zeros(p.shape)

        for i in range(nsteps):
            d = eps*self.D._p_dot_invmass(p)
            sum += d
            absdiff.append(N.mean(N.abs(sum)))
            print "absdiff",i,absdiff[-1]
            q = q + d
            if i<nsteps-1:
                p = p-eps*self.grad_pot(q)

            if self.traceh: 
                H.append(self.pot(q) + self.kinetic(p - eps/2.*self.grad_pot(q)))
                print "step",i,nsteps,eps,H[-1]
        
        # make final half step of momentum
        p = p - eps/2.*self.grad_pot(q)

        if self.traceh:
            H = N.array(H)
            pylab.plot(H)
            print (H.max()-H.min())*1./N.mean(H)
            pylab.show()

        H_next = self.pot(q), self.kinetic(p)

        logp = N.sum(H_current)-N.sum(H_next)
        self.dHam.append(logp)
	self.absdiff = absdiff
        accept_step = False

        if self.verbose:
            print "step",nsteps,eps,"logp",logp

        if logp>=0:
            accept_step=True
        else:
            a = N.exp(logp)
            u = N.random.uniform(0,1)
            if u<a: accept_step=True

        self.count += 1
        if accept_step:
            self.accept_count += 1
            self.q = q.copy()

        if self.verbose:
            print "accept count:",self.accept_count

        self.stepparam.append((nsteps,eps))
        self.times.append(time.time()-t0)
        #self.samples.append(self.q)
        #self.momsamples.append(p)
        self.ham.append(H_next)
        self.acceptflag.append(accept_step)


    def go(self, nloops=100, mask=0):
        """ """
        self.mask = mask

        t0 = time.time()


        for step in range(self.start, nloops):
            self.seteps()
            self.step()
            if not step%10:
                print step,
                sys.stdout.flush()
            self.write_log_txt()
            self.write_output()

        print "\r",

        dt = time.time()-t0

        print ">> done with %i samples"%self.count
        print ">> acceptance rate: ",self.accept_count*1./self.count
        print ">> time: %g sec"%dt
        print ">> time/step: %g sec"%(dt*1./self.count)




def test():
    D = Data()
    H = hmc(x0=0.,dataobj=D)
    H.go(nloops=2)

    pylab.subplot(111,aspect='equal')
    samples = N.transpose(H.samples)
    pylab.hexbin(*samples)
    pylab.show()


if __name__=="__main__":
    test()

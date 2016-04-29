""" Conjugate gradient solver

Ben Granett ben.granett@brera.inaf.it
"""
import sys
import numpy as N
from scipy import sparse
from scipy.optimize import newton_krylov

import fftutils

import time
import cPickle as pickle

nthreads = 4

class Solver:

    def __init__(self, x0=None, gradf=None, A=None, b=None, M=None):
        """ """
        self.gradf = gradf
        self.A = A
        self.b = b
        self.M = M
        self.x0 = x0

        self.lastx = None
        self.counter = 0

    def nonlinear_nr(self, maxloops=10, maxinnerloops=20,
                     eps=1e-3,eps2=1e-5,  reset_freq=20):
        """Newton Raphson step """
        x = self.x0.copy()

        gradf = self.gradf
        hessf = self.Hessf
    
        r = -gradf(self.x0) 
        d = r.copy()
        de_new = N.dot(r,r)
        de_0 = de_new.copy()

        print "de_0",de_0
        lastx = x

        diagnostics = []
        innerloopflag = 0
        counter2 = 0

        t0 = time.time()
        
        for counter in range(int(maxloops)):
            if counter%10==0:
                diagnostics.append((counter,N.max(abs(lastx-x)),de_new,N.max(abs(d)),innerloopflag,counter2,
                                    x.reshape(self.shape)[:,self.shape[1]//2,:]))
                sys.stdout.flush()

            #if counter%100==0
            #    # write out the status
            #    pickle.dump((self.shape,self.length,self.Nbar,0,x.reshape(self.shape),diagnostics),
            #                file(self.diagfile,'w'))

            

            if de_new < eps**2*de_0:
                print " tolerance reached"
                break

            if counter %1==0:
                print time.time()-t0,counter,"diff",N.max(N.abs(x-lastx)),"resid",N.log10(N.max(N.abs(d))),N.log10(de_new),N.log10(eps**2*de_0)

            #print "loop",counter
            de_d = N.dot(d,d)
            lastx = x.copy()

            innerloopflag = -1
            #maxinnerloops = int(counter/50)*50+1

            #ta = time.time()
            sinvd = self.Sinvx(d)
            #print "sinvx time",time.time()-ta

            for counter2 in range(int(maxinnerloops)):

                #ta = time.time()
                a = -N.dot(gradf(x),d)
                #print "    gradtime a",time.time()-ta

                # save a computation of sinv(d) because d does not update
                #Hd = hessf(x,d)
                Hd = sinvd + self.Nbar*self.w*N.exp(x) * d

                a = a/N.dot(d,Hd)

                if not N.all(N.isfinite(a)):
                    print "shit"
                    innerloopflag = -2
                    break

                x = x + a*d/10.

                print "            >",counter2,a,a**2*de_d,N.max(N.abs(a*d))

                if a**2*de_d < eps2**2:
                    innerloopflag = 0
                    break

            #print "    >",innerloopflag,counter2
            rlast = r.copy()
            ta = time.time()
            r = -gradf(x)
            #print "grad time",time.time()-ta
            de_mid = N.dot(r,rlast)
            de_old = de_new

            de_new = N.dot(r,r)
            beta = (de_new-de_mid)/de_old

            #d =r + beta*d
            print "    >",innerloopflag,counter2,de_new,beta
            #sys.stdout.flush()

            #if (counter+1)%reset_freq==0 or N.dot(r,d) <= 0 or beta < 0:
            if (counter+1)%reset_freq==0 or beta < 0:
                print "restarting",counter,N.dot(r,d),beta
                d = r
            else:
                d = r + beta*d


        flag = 0
        if counter>=maxloops-1:
            flag = -1
            print "no convergence after ",maxloops

        return x, flag, diagnostics

    def predict_conv(self, counter,d, stoppingpoint,nfit=10):
        """ """
        try:
            self._convlist.append(d)
        except:
            self._convlist = [d]

        if len(self._convlist)<10:
            i=100
        else:
            nfit = len(self._convlist)//2
            y = N.log(self._convlist[-nfit:])
            x = N.arange(len(y))
            fit = N.polyfit(x,y,1)
            a,b = fit
            i = 1./a*(N.log(stoppingpoint)-b)
            i = int(i-nfit)

        if counter==0:
            i = max(1,i)
        f = int(counter*1./(counter+i)*80)
        print "\r"+" "*(80),

        print "\r%i %i"%(counter,i)+"~"*(f)+"&"+"-"*(80-f),
        sys.stdout.flush()

        return i


    def nonlinear_secant(self, maxloops=1000, maxinnerloops=100,
                  eps=1e-3, eps2=1e-10, sigma0=1e-10, reset_freq=100,callback=None):
        """ """
        x = self.x0.copy()

        gradf = self.gradf
        precond_inv = self.precond_inv
    
        r = -gradf(self.x0) 
        s = precond_inv(r)
        d = s.copy()
        de_new = N.dot(r,d)
        de_0 = de_new.copy()

        stoppingpoint = de_0*eps**2

        diagnostics = []
        innerloopflag = 0
        counter2 = 0


        lastx = x
        for counter in range(int(maxloops)):

            diagnostics.append((counter,N.max(abs(lastx-x)),de_new,N.max(abs(d)),innerloopflag,counter2))
            if not callback==None:
                callback(diagnostics[-1],x)
            
            self.predict_conv(counter,de_new,stoppingpoint)

            if de_new < stoppingpoint:
                print " tolerance reached",counter,de_new
                break

            #if counter %1==0:
            #print "loop",counter,"resid",N.log10(N.max(N.abs(d))),N.log10(de_new), N.log10(eps**2*de_0)

            de_d = N.dot(d,d)
            a = -sigma0


            lastx = x
            innerloopflag = -1
            eta_p = N.dot(gradf(x-a*d), d)
            for counter2 in range(int(maxinnerloops)):
                eta = N.dot(gradf(x), d)
                #print counter2,"   eta =", eta, eta_p
         
                a = a * eta/(eta_p - eta)
                if not N.isfinite(a):
                    innerloopflag = -2
                    break
        

                x = x + a*d
                eta_p = eta

                if a**2*de_d < eps2**2:
                    innerloopflag = 0
                    break

            if counter2>=maxinnerloops-1:
                #print "warning, max inner loops reached",maxinnerloops
                pass

            r = -gradf(x)
            de_old = de_new
            de_mid = N.dot(r,s)

            s = precond_inv(r)
            de_new = N.dot(r,s)

            beta = (de_new-de_mid)/de_old
            if (counter+1)%reset_freq==0 or beta <= 0:
                #print "restarting",counter,beta
                d = s.copy()
            else:
                d = s + beta*d

        flag = 0
        if counter>=maxloops-1:
            flag = -1
            print "> no convergence after ",maxloops

        print "> max residual",N.max(N.abs(self.gradf(x)))

        return x, flag, diagnostics

    def linear_sd(self, maxloops=100, eps=1e-10, reset_freq=10):
        """ Solve Ax=b using the method of steepest decent """
        A = self.A
        b = self.b

        x = self.x0.copy()

        # compute residual (direction of gradient)
        r = b - A*self.x0

        delta = N.dot(r,r)
        
        tolerance = delta*eps**2

        for counter in range(int(maxloops)):
            if delta < tolerance:
                print "error tolerance reached"
                break
            print "loop",counter

            q = A*r
            a = delta/N.dot(r,q)
            x = x + a*r

            if (counter+1) % reset_freq == 0:
                print "resetting"
                r = b - A*x
            else:
                r = r - a*q

            delta = N.dot(r,r)
        
        return x



    def linear_ben(self, maxloops=100, eps=1e-10, reset_freq=10):
        """ solve Ax=b """
        A = self.A
        b = self.b
        M = self.M
        #if M==None: 
        M = sparse.eye(*A.shape)
        
        x = self.x0.copy()
        
        r = b - A*self.x0

        p = M*r.copy()

        rr = N.dot(r,p)

        tolerance = rr*eps**2

        for counter in range(int(maxloops)):
            if rr < tolerance:
                print "error tolerance reached"
                break
            print "loop",counter
        
            Ap = A*p

            a = rr/N.dot(p,Ap)

            x = x + a*p

            if (counter+1) % reset_freq == 0:
                print "resetting"
                r = b - A*x
            else:
                r = r-a*Ap

            s = M*r
            
            rrnew = N.dot(r,s)

            p = s+rrnew/rr*p

            rr = rrnew


        return x

    def linear(self, tol=1e-5,maxiter=500):
        """ """
        y,info = sparse.linalg.bicgstab(self.A, self.b, x0=self.x0,
        #y,info = sparse.linalg.cg(self.A, self.b, x0=self.x0,
                                  maxiter=maxiter,
                                  tol=tol,
                                  callback=self.callback)
        if info != 0:
            print "Warning! cg returned non-zero:",info
        return y

    def linear_brute(self):
        """ """
        Ainv = N.linalg.inv(self.A.todense())
        return N.dot(Ainv,self.b)


    def nonlinear(self, rdiff=1e-10):
        """ """
        r = newton_krylov(self.gradf, self.x0, rdiff=rdiff, verbose=True)
        return r



    def computeSb(self, x):
        """ Compute the product of the signal matrix and a vector
        requires self.cf to be defined.
        """
        out = []
        
        m = len(self.centers)
        for j in range(m/self.batch+1):
            a = j*self.batch
            b = min((j+1)*self.batch, m)

            key = a
            Signal = self.cf.getMatrix(self.centers,self.centers[a:b],key=key).transpose()

            r = Signal*x

            out.append(r)

        return N.concatenate(out)



    def Sinvx(self, x):
        """ Compute the whitening filter dot(inv(S),x)
        """
        #norm = N.prod(self.shape)**.5
        xr = x.reshape(self.shape)

        xk = fftutils.gofft(xr, nthreads=nthreads)
        xout = fftutils.gofftinv(xk*self.signalinv, nthreads=nthreads)

        xout = N.real(xout).flatten()

        return xout

    def fourier_Ax(self,A,x):
        """
        """
        xr = x.reshape(self.shape)

        xk = fftutils.gofft(xr, nthreads=nthreads)
        xout = fftutils.gofftinv(xk*A, nthreads=nthreads)

        xout = N.real(xout).flatten()

        return xout


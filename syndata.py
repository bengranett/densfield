

    import simbox


def syndata():
        k = N.logspace(-2,1,100)
        k,pk = pk_class.pk(pkparams, kk=k)

        pk *= 1.5**2  # bias

        assert(N.all(N.isfinite(pk)))

        step = params.cellw
        shape = weight.shape
        length = N.array(shape)*step

        S = simbox.SimBox(k, pk, shape, length, applywindow=True, cachefile=None,lognorm=True)
        mu = 0.5*N.sum(S.pkgrid)*1./N.prod(length)

        s = S.realize()-mu
        N.save("%s/input_delta.npy"%savedir, s)

        v = 4**3
        data_syn = v*nbar*weight*N.exp(s)
        print N.mean(N.exp(s))
        print "means",data_syn.mean(),data.mean()

        data_syn = N.random.poisson(data_syn)


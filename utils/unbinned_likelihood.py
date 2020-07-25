import numpy                as np
import pandas               as pd

from scipy            import optimize
from scipy            import stats
from scipy.integrate  import quad
from scipy.optimize   import minimize



def ll_func(data, lims):
    E = data
    n = len(E)

    a, b = lims
    c, r = (a+b)/2., 1/(b-a)
    slim = 2./(b-a)**2.

    def ll(x):
        p = stats.poisson.pmf(mu=x[0]+x[1], k=n)*pdf(x)
        constraints = np.array([(p<0).any(), x[0]<0, x[1]<0, x[2]<-slim, x[2]>slim])

        if constraints.any():
            return np.inf
        else:
            return -np.log(p).sum()


    def cov(x):
        mus, mub = x[0], x[1]
        f = mus/(mus+mub)

        p = pdf(x)

        mud, sigd = jac_s(x)
        sb        = jac_b(x)

        musp   =  mub/(mus+mub)**2*(fs(x)-fb(x))/p
        mubp   = -mus/(mus+mub)**2*(fs(x)-fb(x))/p
        sp   = (1-f)*sb   /p
        mup  =  f    *mud /p
        sigp =  f    *sigd/p


        A = np.array([[(musp*musp).sum(), (musp*mubp).sum(), (musp*sp).sum(), (musp*mup).sum(), (musp*sigp).sum()],
                      [(mubp*musp).sum(), (mubp*mubp).sum(), (mubp*sp).sum(), (mubp*mup).sum(), (mubp*sigp).sum()],
                      [  (sp*musp).sum(),   (sp*mubp).sum(),   (sp*sp).sum(),   (sp*mup).sum(),   (sp*sigp).sum()],
                      [ (mup*musp).sum(),  (mup*mubp).sum(),  (mup*sp).sum(),  (mup*mup).sum(),  (mup*sigp).sum()],
                      [(sigp*musp).sum(), (sigp*mubp).sum(), (sigp*sp).sum(), (sigp*mup).sum(), (sigp*sigp).sum()]])

        b = -n/(mus+mub)**2
        B = np.array([[b, b, 0, 0, 0],
                      [b, b, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])

        C = der2(x)

        covinv = -(B -A + C)

        return np.linalg.inv(covinv)


    def der2(x):
        mus, mub = x[0], x[1]
        f = mus/(mus+mub)

        p = pdf(x)

        mud, sigd = jac_s(x)
        sb        = jac_b(x)

        mu2p, musig2p, sig2p = der2_s(x)

        mus2p    =    -2*mub*(fs(x)-fb(x))/(mus+mub)**3
        mub2p    =     2*mus*(fs(x)-fb(x))/(mus+mub)**3
        musmub2p = (mus-mub)*(fs(x)-fb(x))/(mus+mub)**3

        cs =  mub/(mus+mub)**2
        cb = -mus/(mus+mub)**2

        D2 = np.array([[   (mus2p/p).sum(), (musmub2p/p).sum(), -(cs*sb/p).sum(),    cs*(mud/p).sum(),   (cs*sigd/p).sum()],
                       [(musmub2p/p).sum(),    (mub2p/p).sum(), -(cb*sb/p).sum(),    cb*(mud/p).sum(),   (cb*sigd/p).sum()],
                       [  -cs*(sb/p).sum(),   -cb*(sb/p).sum(),                0,                   0,                   0],
                       [  cs*(mud/p).sum(),   cb*(mud/p).sum(),                0,    f*(mu2p/p).sum(), f*(musig2p/p).sum()],
                       [ cs*(sigd/p).sum(),  cb*(sigd/p).sum(),                0, f*(musig2p/p).sum(),   f*(sig2p/p).sum()]])
        return D2

    def pdf(x):
        mus, mub = x[0], x[1]
        return mus/(mus+mub)*fs(x) + mub/(mus+mub)*fb(x)

    def fb(x):
        s = x[2]
        return s*(E-c) + r

    def jac_b(x):
        return E-c

    def fs(x):
        mu, sig = x[3], x[4]
        #A, _ = quad(gauss, a, b, args=(mu, sig))
        A = sig*(2*np.pi)**(1/2.)
        return (1/A)*gauss(E, mu, sig)

    def jac_s(x):
        mu, sig = x[3], x[4]

        muder  = fs(x)*(E-mu)/sig**2
        sigder = fs(x)*(-1/sig + (E-mu)**2/sig**3)

        return np.array([muder, sigder])


    def der2_s(x):

        mu, sig = x[3], x[4]

        muder  = fs(x)*(E-mu)/sig**2
        sigder = fs(x)*(-1/sig + (E-mu)**2/sig**3)

        muder2    = muder*(E-mu)/sig**2 + fs(x)*(-1/sig**2)
        sigder2   = sigder*(-1/sig + (E-mu)**2/sig**3) + fs(x)*(1/sig**2 -3*(E-mu)**2/sig**4)
        musigder2 =  muder*(-1/sig + (E-mu)**2/sig**3) + fs(x)*(-2*(E-mu)/sig**3)

        return np.array([muder2, musigder2, sigder2])


    def gauss(E, mu, sig):
        return np.e**(-(E-mu)**2/(2*sig**2))

    return ll, cov


def pkfit(data, lims, x0):
    '''data: energy values (np.array);
       lims: lower and upper bounds (np.array);
       x0:   fitting guess (np.array)'''
    E=data
    a, b = lims

    ll, cov = ll_func(E, [a, b])
    res = minimize(ll, x0, method='powell',
                    options={'disp':True, 'ftol':1e-15 , 'maxiter':1e2})

    return res, cov(res.x)


def deppdf(E, x, lims):
    '''Double escape peak pdf for x parameters'''
    a, b = lims
    c, r = (a+b)/2., 1/(b-a)

    def pdf(x):
        mus, mub = x[0], x[1]
        return mus/(mus+mub)*fs(x) + mub/(mus+mub)*fb(x)

    def fb(x):
        s = x[2]
        return s*(E-c) + r

    def fs(x):
        mu, sig = x[3], x[4]
        #A, _ = quad(gauss, a, b, args=(mu, sig))
        A = sig*(2*np.pi)**(1/2.)
        return (1/A)*gauss(E, mu, sig)

    def gauss(E, mu, sig):
        return np.e**(-(E-mu)**2/(2*sig**2))

    return pdf(x)

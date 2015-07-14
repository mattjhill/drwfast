import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin

from emcee import EnsembleSampler

from .tridiagonal import snsolve

my_neg_inf = float(-1.0e+300)
my_pos_inf = float( 1.0e+300)

tau_floor     = 1.e-6
tau_ceiling   = 1.e+5
sigma_floor   = 1.e-6
sigma_ceiling = 1.e+2
logtau_floor     = np.log(tau_floor) 
logtau_ceiling   = np.log(tau_ceiling)   
logsigma_floor   = np.log(sigma_floor)  
logsigma_ceiling = np.log(sigma_ceiling) 

def lnlike(sigma, tau, lc, return_chisq=False):
    """
    Calculate the log-likelihood as laid out in Rybicki/Press 95
    """
    L = np.ones(len(lc.t))
    err2 = lc.yerr**2
    var = sigma**2
    
    # get the log determinant of C^-1 and solve 
    # C a = L
    ldetc, a = snsolve(lc.t, err2, var, tau, L)
    # Calculate Cq = 1/(L^T C^-1 L)
    Cq = 1/np.dot(L.T, a)

    # solve C b = y
    _, b = snsolve(lc.t, err2, var, tau, lc.y)
    
    c = np.dot(L.T, b)
    c *= Cq

    _, d = snsolve(lc.t, err2, var, tau, c*L)
    
    chisq = np.dot(lc.y.T, b-d)
    lnl = -.5*chisq + .5*np.log(Cq) -.5*ldetc
    if return_chisq:
        return lnl, chisq

    return lnl

def unpacksinglepar(p):
    """ 
    Internal Function: Unpack the physical parameters from input 1-d array for single mode.
    """
    if p[0] > logsigma_ceiling :
        sigma = sigma_ceiling
    elif p[0] < logsigma_floor :
        sigma = sigma_floor
    else :
        sigma = np.exp(p[0])
    if p[1] > logtau_ceiling :
        tau = tau_ceiling
    elif p[1] < logtau_floor :
        tau = tau_floor
    else :
        tau = np.exp(p[1])

    return (sigma, tau)


def lnprob(p, lc, set_prior=True):
    #sigma, tau = unpacksinglepar(p)
    sigma, tau = np.exp(p[0]), np.exp(p[1])
    lnl = lnlike(sigma, tau, lc)

    prior = 0.0
    if set_prior:
        prior += -np.log(sigma)
        if tau > lc.cont_cad:
            prior += -np.log(tau/lc.cont_cad)
        elif tau < 0.001:
            prior += my_neg_inf
        else:
            prior += -np.log(lc.cont_cad/tau)

    return lnl + prior

class DRWModel(object):
    def __init__(self, lc):
        self.lc = lc

    def do_map(self, pinit, set_prior=False):
        """
        Get the Maximum A Posterior (MAP) DRW parameters.
        """
        func = lambda _p : -lnprob(_p, self.lc, set_prior=set_prior)
        p_best, lnl_best = fmin(func, pinit, full_output=True, disp=False)[:2]
        return p_best, -lnl_best    

    def get_Linf(self, set_prior=False):
        """
        Get Likelihood that tau -> infinity.  
        """
        func = lambda _p : -lnprob([_p, 20], self.lc, set_prior=set_prior)
        p_best, lnl_best = fmin(func, 1, full_output=True, disp=False)[:2]
        return -lnl_best

    def get_Lnoise(self, set_prior=False):
        """
        Get Likelihood that tau -> 0.
        """
        func = lambda _p : -lnprob([_p, -20], self.lc, set_prior=set_prior)
        p_best, lnl_best = fmin(func, 1, full_output=True, disp=False)[:2]
        return -lnl_best        

    def get_chisq(self, sigma, tau):
        lnl, chisq = lnlike(sigma, tau, self.lc, return_chisq=True)
        return chisq/(self.lc.npt - 2)

    def do_mcmc(self, nwalker=100, nburn=50, nchain=50, threads=1, set_prior=True):

        # initial walkers for MCMC
        ndim = 2
        p0 = np.random.rand(nwalker*ndim).reshape(nwalker, ndim)
        p0[:,0] = p0[:,0] - 0.5 + np.log(self.lc.cont_std)
        p0[:,1] = np.log(self.lc.rj*0.5*p0[:,1])

        #start sampling
        sampler = EnsembleSampler(nwalker, ndim, lnprob, args=(self.lc,set_prior), threads=threads)
        # burn-in 
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        sampler.reset()
        # actual samples
        sampler.run_mcmc(pos, nchain, rstate0=state)

        self.flatchain = sampler.flatchain
        self.lnprobability = sampler.lnprobability

    def get_results(self, pcts=[16, 50, 84]):
        hpd = np.zeros((3,2))
        hpd[0] = np.percentile(self.flatchain, pcts[0], axis=0)
        hpd[1] = np.percentile(self.flatchain, pcts[1], axis=0)
        hpd[2] = np.percentile(self.flatchain, pcts[2], axis=0)
        return hpd

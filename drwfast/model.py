import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin

from emcee import EnsembleSampler
from ._tridiagonal import lnlike

class DRWModel(object):
    """ 
    The damped random walk model object 

    Attributes
    ----------
    lc : LightCurve object
        A drwfast light curve to model
    t : array_like
        Array containing the observation times
    y : array_like
        Array containing the observations
    yerr : array_like
        Array containg the error bars for the observations
    err2 : array_like
        The squared errors, which will be passed to the likelihood function

    """
    def __init__(self, lc):
        self.lc = lc
        self.t = self.lc.t
        self.y = self.lc.y
        self.yerr = self.lc.yerr
        self.err2 = self.yerr**2

    def lnprob(self, p):
        """
        The log-probability, i.e. the log-likelihood plus the prior distribution.

        Parameters
        ----------
        p : array_like
            an array containing the the parameters at which to evaluate the 
            log-probability, i.e. [log_sigma, log_tau]

        Returns
        -------
        lnp : float
            The log-probability
        """
        #sigma, tau = unpacksinglepar(p)
        sigma, tau = np.exp(p[0]), np.exp(p[1])
        var = sigma**2
        lnl = lnlike(var, tau, self.t, self.y, self.err2)
        if np.isnan(lnl):
            return -np.inf
        else:
            return lnl
        # prior = 0.0
        # if set_prior:
        #     prior += -np.log(sigma)
        #     if tau > lc.cont_cad:
        #         prior += -np.log(tau/lc.cont_cad)
        #     elif tau < 0.001:
        #         prior += my_neg_inf
        #     else:
        #         prior += -np.log(lc.cont_cad/tau)

        # return lnl + prior

    def do_map(self, pinit):
        """
        Get the Maximum A Posterior (MAP) DRW parameters.
        """
        func = lambda _p : -self.lnprob(_p)
        p_best, lnl_best = fmin(func, pinit, full_output=True, disp=False)[:2]
        return p_best, -lnl_best    

    def do_mcmc(self, nwalker=100, nburn=50, nchain=50, threads=1, set_prior=True):
        """
        Find the best fitting parameters for the light curve via Monte Carlo 
        Markov Chain (MCMC) using the EnsembleSampler from emcee

        Parameters
        ----------
        nwalker : int
            the number walkers
        nburn : int
            the number of steps in the burn-in phase
        nchain : int
            the number of steps in the actual sampling phase
        threads: int
            number of threads to use 
        set_prior : bool
            whether to use a log-prior (True) or an empty prior (False)

        Notes
        -----
        The chain itself and the log-probability at each step in the chain
        can be accessed via ``flatchain`` and ``lnprobability`` after the chain
        has been run.

        """

        # initial walkers for MCMC
        ndim = 2
        p0 = np.random.rand(nwalker*ndim).reshape(nwalker, ndim)
        p0[:,0] = p0[:,0] - 0.5 + np.log(self.lc.cont_std)
        p0[:,1] = np.log(self.lc.rj*0.5*p0[:,1])

        #start sampling
        sampler = EnsembleSampler(nwalker, ndim, self.lnprob, threads=threads)
        # burn-in 
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        sampler.reset()
        # actual samples
        sampler.run_mcmc(pos, nchain, rstate0=state)

        self.flatchain = sampler.flatchain
        self.lnprobability = sampler.lnprobability

    def get_results(self, pcts=[16, 50, 84]):
        """
        Get the results of the MCMC sampling

        Parameters
        ----------
        pcts : list
            three perecentiles of the posterior, defaults to +/- 1 sigma
            and the median.

        Returns
        -------
        hpd : array_like
            a 3x2 array containing the specified percentiles for log_sigma
            and log_tau

        """
        hpd = np.zeros((3,2))
        hpd[0] = np.percentile(self.flatchain, pcts[0], axis=0)
        hpd[1] = np.percentile(self.flatchain, pcts[1], axis=0)
        hpd[2] = np.percentile(self.flatchain, pcts[2], axis=0)
        return hpd

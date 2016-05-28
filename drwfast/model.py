import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin

from emcee import EnsembleSampler

from .tridiagonal import snsolve
from .grp import GRP
from .lightcurve import LightCurve

my_neg_inf = float(-1.0e+300)
my_pos_inf = float( 1.0e+300)



class DRWModel(object):
    """ The damped random walk model object 

        :param lc:
            A lightcurve object
    """
    def __init__(self, lc):
        self.lc = lc
    
    def lnlike(self, sigmasqr, tau, lc, return_chisq=False):
        """
        Calculate the log-likelihood as laid out in Rybicki/Press 95
        """
        L = np.ones(len(lc.t))
        err2 = lc.yerr**2
        var = sigmasqr*tau/2
        
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
        if ~np.isfinite(lnl):
            lnl = -np.inf
            
        if return_chisq:
            return lnl, chisq

        return lnl

    def lnprob(self, p, lc, set_prior=True, sigmasqr_min=1e-10):
        #sigma, tau = unpacksinglepar(p)
        sigmasqr, tau = 10**(p[0]), 10**(p[1])
        lnl = self.lnlike(sigmasqr, tau, lc)

        if set_prior:
            if tau > lc.dt_tot*10:
                return -np.inf
            if tau < lc.dt_min/10:
                return -np.inf
            if sigmasqr < sigmasqr_min:
                return -np.inf

        return lnl

    def do_map(self, pinit, set_prior=False):
        """
        Get the Maximum A Posterior (MAP) DRW parameters.
        """
        func = lambda _p : -self.lnprob(_p, self.lc, set_prior=set_prior)
        self.p_map, lnl_best = fmin(func, pinit, full_output=True, disp=False)[:2]
        return self.p_map, -lnl_best    

    def get_Linf(self, set_prior=False):
        """
        Get Likelihood that tau -> infinity.  
        """
        logtau = np.log10(self.lc.dt_tot*1000)
        func = lambda _p : -self.lnprob([_p, logtau], self.lc, set_prior=set_prior)
        p_best, lnl_best = fmin(func, 1, full_output=True, disp=False)[:2]
        return -lnl_best

    def get_Lnoise(self, set_prior=False):
        """
        Get Likelihood that tau -> 0.
        """
        logtau = np.log10(self.lc.dt_min/1000)
        func = lambda _p : -self.lnprob([_p, logtau], self.lc, set_prior=set_prior)
        p_best, lnl_best = fmin(func, 1, full_output=True, disp=False)[:2]
        return -lnl_best        

    def get_chisq(self, sigma, tau):
        lnl, chisq = self.lnlike(sigma, tau, self.lc, return_chisq=True)
        return chisq/(self.lc.npt - 2)

    def do_mcmc(self, nwalker=100, nburn=50, nchain=50, threads=1, set_prior=True):

        # initial walkers for MCMC
        ndim = 2
        pinit = np.zeros((nwalker, ndim))
        pinit[:,0] = np.random.uniform(-10, -2, nwalker)
        pinit[:,1] = np.random.uniform(np.log10(self.lc.dt_min/10), np.log10(self.lc.dt_tot*10), nwalker)

        #start sampling
        sampler = EnsembleSampler(nwalker, ndim, self.lnprob, args=(self.lc,set_prior), threads=threads)
        # burn-in 
        pos, prob, state = sampler.run_mcmc(pinit, nburn)
        sampler.reset()
        # actual samples
        sampler.run_mcmc(pos, nchain, rstate0=state)
        self.sampler = sampler
        self.flatchain = sampler.flatchain
        self.lnprobability = sampler.lnprobability

    def predict(self, t_pred, p):
        sigmasqr, tau = 10**(p[0]), 10**(p[1])
        K = sigmasqr*tau/2*np.exp(-abs(np.subtract.outer(t_pred,self.lc.t))/tau)
        C = sigmasqr*tau/2*np.exp(-abs(np.subtract.outer(self.lc.t, self.lc.t))) + np.diag(self.lc.yerr**2)
        mu = np.dot(K, np.linalg.solve(C, self.lc.y))
        K_pred = sigmasqr*tau/2*np.exp(-abs(np.subtract.outer(t_pred,t_pred))/tau)
        cov = K_pred - np.dot(K, np.linalg.solve(C, K.T))
        return mu, cov

    def get_results(self, pcts=[16, 50, 84]):
        hpd = np.zeros((3,2))
        hpd[0] = np.percentile(self.flatchain, pcts[0], axis=0)
        hpd[1] = np.percentile(self.flatchain, pcts[1], axis=0)
        hpd[2] = np.percentile(self.flatchain, pcts[2], axis=0)
        return hpd

    def compute_sigma_level(self, trace1, trace2, nbins=50):
        """From a set of traces, bin by number of standard deviations"""
        L, xbins, ybins = np.histogram2d(trace1, trace2, nbins)
        L[L == 0] = 1E-16
        logL = np.log(L)

        shape = L.shape
        L = L.ravel()

        # obtain the indices to sort and unsort the flattened array
        i_sort = np.argsort(L)[::-1]
        i_unsort = np.argsort(i_sort)

        L_cumsum = L[i_sort].cumsum()
        L_cumsum /= L_cumsum[-1]
        
        xbins = 0.5 * (xbins[1:] + xbins[:-1])
        ybins = 0.5 * (ybins[1:] + ybins[:-1])

        return xbins, ybins, L_cumsum[i_unsort].reshape(shape)

    def plot_MCMC_trace(self, ax, trace, scatter=False, **kwargs):
        """Plot traces and contours"""
        xbins, ybins, sigma = self.compute_sigma_level(trace[1], trace[0])
        if scatter:
            ax.plot(trace[1], trace[0], ',k', ms=.1, zorder=1, alpha=0.7, rasterized=True)
        ax.contour(xbins, ybins, sigma.T, levels=[.68, 0.95], linewidths=0.25, **kwargs)

    def plot_posterior(self):
        figsize = plt.rcParams['figure.figsize']
        figsize[1] 
        fig = plt.figure(figsize=(figsize[0], figsize[1]*2))
        # joint distribution
        axTS = fig.add_axes([0.2, .75, .75, .2])
        axJ = fig.add_axes([0.2, 0.1, 0.5, 0.325])               # [left, bottom, width, height]
        # side histogram
        axY = fig.add_axes([0.7, 0.1, 0.25, 0.325], sharey=axJ) 
        # top histogram
        axX = fig.add_axes([0.2, 0.425, 0.5, 0.2], sharex=axJ) 
        #fig = plt.figure(figsize=(10,8))
        #ax = gridspec.GridSpec(3, 3, width_ratios=[3, 1], sharex='col', sharey='row',) 
        self.plot_MCMC_trace(axJ, self.flatchain.T, scatter=True, colors='r')
        nX, binsX, _ = axX.hist(self.flatchain[:,1], bins=100, histtype='step', color='k', linewidth=0.5)
        nY, binsY, _ = axY.hist(self.flatchain[:,0], bins=100, histtype='step', color='k', linewidth=0.5,
                              orientation='horizontal')

        axTS.errorbar(self.lc.t, self.lc.y, self.lc.yerr, fmt='sk', ms=2)
        axTS.set_xlabel("time")
        axX.yaxis.set_visible(False)
        axX.xaxis.set_visible(False)
        axY.xaxis.set_visible(False)
        axY.yaxis.set_visible(False)


class SupOUModel(object):
    """ The damped random walk model object 

        :param lc:
            A lightcurve object
    """
    def __init__(self, lc):
        self.lc = lc

    def set_weights(self, lc, tau_H, alpha, M=30):
        log_omega = np.zeros(M)
        tau_L = lc.dt_tot*10
        # for i in range(M):
        #     log_omega[i] = np.log10(1/tau_L) + float((i))/(M-1)*(np.log10(1/tau_H) - np.log10(1/tau_L))

        # omega = 10**(log_omega)
        omega = np.exp(np.log(1/tau_L) + (np.arange(1., M+1)-1)/(M-1)*(np.log(1/tau_H) - np.log(1/tau_L)))
        weights = omega**(1-alpha/2)/np.sqrt((omega**(2-alpha)).sum())
        tau = 1/omega
        return weights, tau

    def lnlike(self, sigmasqr, tau_H, lc, return_chisq=False):
        """
        Calculate the log-likelihood as laid out in Rybicki/Press 95
        """

        weights, tau = self.set_weights(lc, tau_H, alpha=1.0)
        diagonal = (sigmasqr*weights**2*tau/2).sum() + lc.yerr**2
        grp = GRP(sigmasqr*weights**2*tau/2, 1/tau, lc.t, diagonal)
        grp.assemble_matrix()
        grp.factor()

        L = np.ones(len(lc.t))
        err2 = lc.yerr**2
        
        # get the log determinant of C^-1 and solve 
        # C a = L
        ldetc = grp.logdeterminant()
        a = grp.solve(L)
        # Calculate Cq = 1/(L^T C^-1 L)
        Cq = 1/np.dot(L.T, a)

        # solve C b = y
        b = grp.solve(lc.y)
        
        c = np.dot(L.T, b)
        c *= Cq

        d = grp.solve(c*L)
        
        chisq = np.dot(lc.y.T, b-d)
        lnl = -.5*chisq + .5*np.log(Cq) -.5*ldetc
        if ~np.isfinite(lnl):
            lnl = -np.inf
            
        if return_chisq:
            return lnl, chisq

        return lnl

    def lnprob(self, p, lc, set_prior=True, sigmasqr_min=1e-10):
        #sigma, tau = unpacksinglepar(p)
        sigmasqr, tau_H = 10**(p[0]), 10**(p[1])
        lnl = self.lnlike(sigmasqr, tau_H, lc)

        if set_prior:
            if tau_H > lc.dt_tot*10:
                return -np.inf
            if tau_H < lc.dt_min/10:
                return -np.inf
            if sigmasqr < sigmasqr_min:
                return -np.inf

        return lnl

    def do_map(self, pinit, set_prior=False):
        """
        Get the Maximum A Posterior (MAP) DRW parameters.
        """
        func = lambda _p : -self.lnprob(_p, self.lc, set_prior=set_prior)
        self.p_map, lnl_best = fmin(func, pinit, full_output=True, disp=False)[:2]
        return self.p_map, -lnl_best 

    def do_mcmc(self, nwalker=100, nburn=50, nchain=50, threads=1, set_prior=True):

        # initial walkers for MCMC
        ndim = 2
        pinit = np.zeros((nwalker, ndim))
        pinit[:,0] = np.random.uniform(-10, -2, nwalker)
        pinit[:,1] = np.random.uniform(np.log10(self.lc.dt_min/10), np.log10(self.lc.dt_tot*10), nwalker)

        #start sampling
        sampler = EnsembleSampler(nwalker, ndim, self.lnprob, args=(self.lc,set_prior), threads=threads)
        # burn-in 
        pos, prob, state = sampler.run_mcmc(pinit, nburn)
        sampler.reset()
        # actual samples
        sampler.run_mcmc(pos, nchain, rstate0=state)
        self.sampler = sampler
        self.flatchain = sampler.flatchain
        self.lnprobability = sampler.lnprobability
        
    def sample(self, lc, sigmasqr, tau_H, alpha):
        weights, tau = self.set_weights(lc, tau_H, alpha)
        dt = np.subtract.outer(lc.t, lc.t)
        C = np.zeros_like(dt) + np.diag(lc.yerr**2)
        for i in range(len(tau)):
            C += sigmasqr/2*weights[i]**2*tau[i]*np.exp(-abs(dt)/tau[i])
        
        y = np.random.multivariate_normal(np.zeros_like(lc.t), C)
        return LightCurve(np.column_stack([lc.t, y, lc.yerr]))

    def psd(self, f, sigmasqr, tau_H, alpha=1.0):
        tau_L = self.lc.dt_tot*10
        weights, tau = self.set_weights(self.lc, tau_H, alpha=alpha)
        P = np.zeros(len(f))
        for i in range(len(tau)):
            P += 2*weights[i]**2*tau[i]**2 / (1 + (2*np.pi*tau[i]*f)**2)
        P *= sigmasqr
        return P

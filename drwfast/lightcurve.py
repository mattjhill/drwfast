import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic

def make_lc(fname):
	data = np.loadtxt(fname)
	return LightCurve(data)


class LightCurve(object):
	""" A light curve object"""
	
	def __init__(self, data):
		data = data[data[:,0].argsort()]
		self.npt = len(data)
		self.t = data[:,0]
		self.y = data[:,1]
		self.yerr = data[:,2]
		self.rj = self.t[-1] - self.t[0]
		self.cont_std = np.std(self.y)
		self.cont_cad_arr  = self.t[1:] - self.t[:-1]
		self.cont_cad = np.median(self.cont_cad_arr)

	def errorbar(self):
		"""
		plot the light curve
		"""

		plt.errorbar(self.t, self.y, self.yerr, fmt='o')

	def structure_function(self):
		"""
		compute the structure function of the light curve at given time lags
		"""
		dt =  np.subtract.outer(self.t,self.t)[np.tril_indices(self.t.shape[0], k=-1)]
		dm =  np.subtract.outer(self.y,self.y)[np.tril_indices(self.y.shape[0], k=-1)]
		sqrsum, bins, _ = binned_statistic(dt, dm**2, statistic='sum')
		n, _, _ = binned_statistic(dt, dm**2, statistic='count')
		SF = np.sqrt(sqrsum/n)
		lags = 0.5*(bins[1:] + bins[:-1])

		return lags, SF

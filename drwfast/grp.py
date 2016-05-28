import scipy.sparse as sp
from scipy.linalg import solve_banded
from scipy.sparse.linalg import splu
import numpy as np
from numba import jit

@jit(cache=True)
def assemble_matrix_fast(alpha, beta, d, t, m, N):
	""" try to assemble without appending list """

	gamma = np.empty((m, N-1), dtype=np.float)
	for i in range(m):
		for j in range(N-1):
			gamma[i,j]	=	np.exp(-beta[i]*abs(t[j]-t[j+1]))

	twom = 2*m
	blocknnz = 6*m+1
	nBlockSize = twom+1
	M = N*nBlockSize-twom
	nnz = (N-1)*blocknnz+(N-2)*twom+1

	nBlockStart = np.empty(N, dtype=int)
	for k in range(N):
		nBlockStart[k] = k*nBlockSize

	row = np.empty(nnz, dtype=np.float)
	col = np.empty(nnz, dtype=np.float)
	data = np.empty(nnz, dtype=np.float)
	count = 0
	for nBlock in range(N-1):
	#	The starting row and column for the blocks.
	#	Assemble the diagonal first.
		row[count] = nBlockStart[nBlock] 
		col[count] = nBlockStart[nBlock]
		data[count] = d[nBlock]
		count += 1
		for k in range(m):
			row[count] = nBlockStart[nBlock]+k+1
			col[count] = nBlockStart[nBlock]
			data[count] = gamma[k,nBlock]
			count += 1

			row[count] = nBlockStart[nBlock] 
			col[count] = nBlockStart[nBlock]+k+1
			data[count] = gamma[k,nBlock]
			count += 1

			row[count] = nBlockStart[nBlock]+m+k+1 
			col[count] = nBlockStart[nBlock]+twom+1
			data[count] = alpha[k]
			count += 1

			row[count] = nBlockStart[nBlock]+twom+1
			col[count] = nBlockStart[nBlock]+m+k+1
			data[count] = alpha[k]
			count += 1

			row[count] = nBlockStart[nBlock]+k+1
			col[count] = nBlockStart[nBlock]+k+m+1
			data[count] = -1.0
			count += 1

			row[count] = nBlockStart[nBlock]+k+m+1
			col[count] = nBlockStart[nBlock]+k+1
			data[count] = -1.0
			count += 1

	row[count] = M-1
	col[count] = M-1
	data[count] = d[N-1]
	count += 1

	# Assebmles the supersuperdiagonal identity blocks.
	for nBlock in range(N-2):
		for k in range(m):
			row[count] = nBlockStart[nBlock]+m+k+1 
			col[count] = nBlockStart[nBlock]+twom+k+2
			data[count] = gamma[k,nBlock+1]
			count += 1

			row[count] = nBlockStart[nBlock]+twom+k+2
			col[count] = nBlockStart[nBlock]+m+k+1
			data[count] = gamma[k,nBlock+1]
			count += 1

	Aex = sp.csc_matrix((data, (row, col)), shape=(M,M))
	return Aex

class GRP(object):
	def __init__(self, alpha, beta, t, d):
		self.alpha = alpha
		self.beta = beta
		self.t = t
		self.d = d

		self.N = len(t)
		self.m = len(alpha)
		self.nBlockSize = 2*self.m + 1
		self.M = self.N*self.nBlockSize-2*self.m

	def assemble_matrix_slow(self):
		self.gamma = np.empty((self.m,self.N-1))
		for i in range(self.m):
			for j in range(self.N-1):
				self.gamma[i,j]	=	np.exp(-self.beta[i]*abs(self.t[j]-self.t[j+1]))

		twom = 2*self.m
		self.blocknnz = 6*self.m+1
		self.nBlockSize = twom+1
		self.M = self.N*self.nBlockSize-twom
		self.nnz = (self.N-1)*self.blocknnz+(self.N-2)*twom+1

		self.nBlockStart = []
		for k in range(self.N):
			self.nBlockStart.append(k*self.nBlockSize)

		self.row, self.col, self.data = [], [], []
		for nBlock in range(self.N-1):
		#	The starting row and column for the blocks.
		#	Assemble the diagonal first.
			self.row.append(self.nBlockStart[nBlock]) 
			self.col.append(self.nBlockStart[nBlock])
			self.data.append(self.d[nBlock])
			for k in range(self.m):
				self.row.append(self.nBlockStart[nBlock]+k+1) 
				self.col.append(self.nBlockStart[nBlock])
				self.data.append(self.gamma[k,nBlock])

				self.row.append(self.nBlockStart[nBlock]) 
				self.col.append(self.nBlockStart[nBlock]+k+1)
				self.data.append(self.gamma[k,nBlock])

				self.row.append(self.nBlockStart[nBlock]+self.m+k+1) 
				self.col.append(self.nBlockStart[nBlock]+twom+1)
				self.data.append(self.alpha[k])

				self.row.append(self.nBlockStart[nBlock]+twom+1) 
				self.col.append(self.nBlockStart[nBlock]+self.m+k+1)
				self.data.append(self.alpha[k])

				self.row.append(self.nBlockStart[nBlock]+k+1) 
				self.col.append(self.nBlockStart[nBlock]+k+self.m+1)
				self.data.append(-1.0)

				self.row.append(self.nBlockStart[nBlock]+k+self.m+1) 
				self.col.append(self.nBlockStart[nBlock]+k+1)
				self.data.append(-1.0)

		self.row.append(self.M-1)
		self.col.append(self.M-1)
		self.data.append(self.d[self.N-1])

		# Assebmles the supersuperdiagonal identity blocks.
		for nBlock in range(self.N-2):
			for k in range(self.m):
				self.row.append(self.nBlockStart[nBlock]+self.m+k+1) 
				self.col.append(self.nBlockStart[nBlock]+twom+k+2)
				self.data.append(self.gamma[k,nBlock+1])

				self.row.append(self.nBlockStart[nBlock]+twom+k+2) 
				self.col.append(self.nBlockStart[nBlock]+self.m+k+1)
				self.data.append(self.gamma[k,nBlock+1])

		self.Aex = sp.csc_matrix((np.array(self.data), (np.array(self.row), np.array(self.col))), shape=(self.M,self.M))

	def assemble_matrix(self):
		self.Aex = assemble_matrix_fast(self.alpha, self.beta, self.d, self.t, self.m, self.N)


	def factor(self):
		self.factorize = splu(self.Aex)

	def solve(self, rhs):
		rhsex = np.zeros(self.M)
		rhsex[::self.nBlockSize] = rhs
		solex = self.factorize.solve(rhsex)

		return solex[::self.nBlockSize]

	def solve_banded(self, rhs):
		rhsex = np.zeros(self.M)
		rhsex[::self.nBlockSize] = rhs
		if not hasattr(self, 'Aex_diag'):
			Aex_diag = sp.dia_matrix(self.Aex)
		solex = solve_banded((self.m + 1,self.m + 1), Aex_diag.data[::-1,:], rhsex)
		return solex[::self.nBlockSize]


	def logdeterminant(self):
		return np.log(abs(self.factorize.U.diagonal())).sum()

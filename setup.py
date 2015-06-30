from numpy.distutils.core import Extension, setup

setup(name='drwfast',
	packages=["drwfast"],
       ext_modules=[Extension(name='drwfast.tridiagonal', sources=['./drwfast/tridiagonal.f90'])],
       )
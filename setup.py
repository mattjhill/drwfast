from distutils.core import setup, Extension
import numpy

setup(name='tri_utils',
	packages=['tri_utils'],
    ext_modules=[Extension(name="tri_utils._tridiagonal", sources=["./tri_utils/_tridiagonal.c", "./tri_utils/tridiagonal.c"])],
    include_dirs=[numpy.get_include()],
)
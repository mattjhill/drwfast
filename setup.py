from distutils.core import setup, Extension
import numpy

setup(name='drwfast',
	packages=['drwfast'],
    ext_modules=[Extension(name="drwfast._tridiagonal", sources=["./drwfast/_tridiagonal.c", "./drwfast/tridiagonal.c"])],
    include_dirs=[numpy.get_include()],
)
from distutils.core import setup
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy

Options.annotate =True
setup(ext_modules = cythonize('crossPresModels.pyx', annotate=True), include_dirs=[numpy.get_include()])
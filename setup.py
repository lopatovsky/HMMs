from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='cthmm',
    version='0.1',
    description='HMM and CT-HMM library',
    author='Lukas Lopatovsky',
    author_email='lopatovsky@gmail.com',
    license='GPL',
    url='https://github.com/lopatovsky/CT-HMM',
    py_modules=['hmm','cthmm'],
    ext_modules=cythonize('cthmm/*.pyx', language_level=3, include_dirs=[numpy.get_include()]),
    include_dirs=[numpy.get_include()],
    install_requires=[
        'Cython',
        'NumPy',
    ],
    setup_requires=['pytest-runner',],
    tests_require=['pytest',],
)

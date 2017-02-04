from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='hmms',
    version='0.1',
    description='DT-HMM and CT-HMM library',
    author='Lukas Lopatovsky',
    author_email='lopatovsky@gmail.com',
    license='GPL',
    url='https://github.com/lopatovsky/CT-HMM',
    py_modules=['dthmm','cthmm','hmm'],
    ext_modules=cythonize('hmms/*.pyx', language_level=3, include_dirs=[numpy.get_include()]),
    #TODO test: extra_compile_args=['-O3']
    include_dirs=[numpy.get_include()],
    install_requires=[
        'Cython',
        'NumPy',
    ],
    setup_requires=['pytest-runner',],
    tests_require=['pytest',],
)

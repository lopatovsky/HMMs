from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy


with open('README') as f:
    long_description = ''.join(f.readlines())


setup(
    name='hmms',
    version='0.1.0.0',
    description='DT-HMM and CT-HMM library',
    author='Lukas Lopatovsky',
    author_email='lopatovsky@gmail.com',
    license='Public Domain',
    keywords='Hidden Markov Model,Continuous-time Hidden Markov Model,HMM,CT-HMM,DT-HMM',
    url='https://github.com/lopatovsky/CT-HMM',
    packages=find_packages(),
    zip_safe=False,
    ext_modules=cythonize('hmms/*.pyx', language_level=3, include_dirs=[numpy.get_include()]),
    #extra_compile_args=['-O3'],
    include_dirs=[numpy.get_include()],
    install_requires=[
        'Cython',
        'NumPy',
        'ipython',
        'matplotlib',
        'pandas',
        'scipy',
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: Public Domain',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Cython',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    setup_requires=['pytest-runner',],
    tests_require=['pytest',],
)

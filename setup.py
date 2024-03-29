import glob
from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy


with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='hmms',
    version='0.2.3',
    description='Efficient discrete and continuous-time hidden Markov model library able to handle hundreds of hidden states',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Lukas Lopatovsky',
    author_email='lopatovsky@gmail.com',
    license='Public Domain',
    keywords='Hidden Markov Model,Continuous-time Hidden Markov Model,HMM,CT-HMM,DT-HMM',
    url='https://github.com/lopatovsky/CT-HMM',
    packages=find_packages(),
    zip_safe=False,
    ext_modules=cythonize(glob.glob('hmms/*.pyx'), language_level=3),
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

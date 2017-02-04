#import pyximport; pyximport.install()
from hmms import hmm
from hmms cimport hmm
import numpy
import scipy.linalg
cimport numpy
cimport cython

#cython: wraparound=False
#cython: boundscheck=False
#cython: nonecheck=False

cdef class CtHMM(hmm.HMM):

    def __init__(self):
        pass

    def meow(self):
        """Make the CTHMM to continuosly meow"""
        print('meooooooow!')

def main():
    my_hmm = CtHMM()
    my_hmm.meow()

    #import scipy.linalg
    #scipy.linalg.expm(x)
    #numpy.linalg.matrix_power(X,3)


    #print( hmm.foo(2) )

    print( type( hmm.HMM ) )


if __name__ == "__main__":
    main()

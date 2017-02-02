#import pyximport; pyximport.install()
from hmms import hmm
from hmms cimport hmm
import numpy
cimport numpy
cimport cython


cdef class HMM:
    def __init__(self):
       pass

cdef class CtHMM(hmm.HMM):

    def __init__(self):
        pass

    def meow(self):
        """Make the CTHMM to continuosly meow"""
        print('meooooooow!')


class Test2(hmm.Test):
    def __init__(self):
        pass

def main():
    my_hmm = CtHMM()
    my_hmm.meow()

    test = hmm.Test()

    print( hmm.foo(2) )

    print( type( hmm.HMM ) )
    print( type( hmm.Test ) )


if __name__ == "__main__":
    main()

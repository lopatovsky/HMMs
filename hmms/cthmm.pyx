import numpy
#import dthmm
cimport numpy
cimport cython

cdef class CtHMM:

    def __init__(self):
        pass

    def meow(self):
        """Make the CTHMM to continuosly meow"""
        print('meooooooow!')


def main():
    my_hmm = CtHMM()
    my_hmm.meow()


if __name__ == "__main__":
    main()

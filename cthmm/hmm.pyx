import numpy
import random
cimport numpy
cimport cython


ctypedef numpy.float64_t DTYPE_t

cdef class HMM:

    cdef numpy.ndarray _a
    cdef numpy.ndarray _b
    cdef numpy.ndarray _pi

    def __init__(self, A, B, Pi):
        """Initialize the HMM by small random values."""
        self._a = A
        self._b = B
        self._pi = Pi
        print("hello init")

    def generate(self, length ):
        """Randomly generate a sequence of states and emissions from parameters."""
        states = numpy.zeros(length)
        emissions = numpy.zeros(length)
        current_state = numpy.random.choice( self._pi.shape[0], 1, p= self._pi)
        for i in range(length):
            states[i] = current_state
            emissions[i] =  numpy.random.choice( self._b.shape[1],1, p = self._b[ current_state,:].flatten() )
            current_state = numpy.random.choice( self._a.shape[1],1, p = self._a[ current_state,:].flatten() )
        return (states, emissions )

    #@classmethod
    def from_parameters( self, A, B, Pi):
        """Initialize the HMM by giving parameters - matrices A,B and vector Pi."""
        self._A = A
        self._B = B
        self._Pi = Pi
        #init_matrices(A,B,Pi)

    cpdef viterbi(self):
        pass


    def from_file( self,file_path ):
        """Initialize the HMM by the file from the file_path storing the parameters A,B,Pi""" ##TODO define the file format.
        print("hello file")

    def _init__(self,num):
        print(num)

    def meow(self):
        """Make the HMM to meow"""
        print('meow!')


def main():
    #my_hmm = HMM()
    A = numpy.array([[0.9,0.1],[0.4,0.6]])
    B = numpy.array([[0.9,0.1],[0.2,0.8]])
    pi = numpy.array( [0.8,0.2] )
    hmm = HMM(A,B,pi)
    (s,e) = hmm.generate(15)
    print(s)
    print(e)

    #hmm2 = HMM.from_parameters(A,B,pi)
    #hmm2 = HMM.from_file("x.hmm")
    #my_hmm.meow()

if __name__ == "__main__":
    main()

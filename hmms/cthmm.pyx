#import pyximport; pyximport.install()
from hmms import hmm
from hmms cimport hmm
import numpy
import scipy.linalg
cimport numpy
cimport cython

import random

#cython: wraparound=False
#cython: boundscheck=False
#cython: nonecheck=False

ctypedef numpy.float64_t float_t
ctypedef numpy.int_t int_t


cdef class CtHMM(hmm.HMM):

    cdef numpy.ndarray _q
    cdef numpy.ndarray _logb
    cdef numpy.ndarray _logpi

    @property
    def b(self):
        return numpy.exp( self._logb )

    @property
    def pi(self):
        return numpy.exp( self._logpi )

    def __init__(self, Q,B,Pi):
        """Initialize the DtHMM by given parameters."""
        self.set_params( Q,B,Pi )

    def set_params( self, Q, B, Pi):
        """Set parameters as their logs to avoid underflow"""
        self._q = Q
        self._logb = numpy.log(B)
        self._logpi = numpy.log(Pi)

    @classmethod
    def random( cls, s, o ):
        """Initialize the class by random parameters of 's' hidden states and 'o' output variables"""
        return cls( *CtHMM.get_random_params( s, o ) )

    @staticmethod
    def get_random_vector( s ):
        """Generate random vector of size (s), with all values summing to one"""
        vec = numpy.random.random(s)
        return vec / numpy.sum(vec)

    @staticmethod
    def get_random_params( s, o ):
        """Generate random parameters A,B and Pi, for number of hidden states (s) and output variables (o)"""

        q = numpy.empty( [s,s] )
        b = numpy.empty( [s,o] )
        pi = numpy.empty( s )

        for i in range( s ):

            ##TODO - do we want for ij i!= j sum to 1?
            vec = CtHMM.get_random_vector(s-1)
            q[i,:i] = vec[:i]
            q[i,i+1:] = vec[i:]
            q[i,i] = -1*numpy.sum(vec)

        for i in range( o ):
            b[i,:] = CtHMM.get_random_vector(o)
        pi = CtHMM.get_random_vector(s)

        return(q,b,pi)



    def generate(self, size ):
        """Randomly generate a sequence of states and emissions from model parameters."""
        q = numpy.array( self._q )
        qt = numpy.empty( q.shape )

        b = numpy.exp( self._logb )
        pi = numpy.exp( self._logpi )

        states = numpy.empty(size,dtype=int)
        emissions = numpy.empty(size,dtype=int)
        times = numpy.empty(size,dtype=int)

        current_state = numpy.random.choice( pi.shape[0], 1, p= pi)
        current_time = 0;
        b = numpy.array( [[0.9,0.1],[0.1,0.9]] )

        for i in range(size):
            states[i] = current_state
            times[i] =  current_time
            emissions[i] =  numpy.random.choice( b.shape[1],1, p = b[ current_state,:].flatten() )

            #observation times will have exponential distances.
            time_interval = int( random.expovariate(0.5) ) + 1
            current_time += time_interval

            #print( time_interval )
            #print(q)
            #print( scipy.linalg.expm(q * time_interval ) )

            qt = scipy.linalg.expm(q * time_interval )

            current_state = numpy.random.choice( qt.shape[1],1, p = qt[ current_state,:].flatten() )

        return ( states, times, emissions )


    cpdef numpy.ndarray[float_t, ndim=2] forward(self, numpy.ndarray[int_t, ndim=1] emissions):
        """From emission sequence calculate the forward variables (alpha) given model parameters.
           Return logaritmus of probabilities.
        """
        cdef numpy.ndarray[float_t, ndim=2] loga = self._loga
        cdef numpy.ndarray[float_t, ndim=2] logb = self._logb
        cdef numpy.ndarray[float_t, ndim=1] logpi = self._logpi
        cdef int i, s, size, states_num,

        size = emissions.shape[0]
        states_num = self._loga.shape[0]
        cdef numpy.ndarray[float_t, ndim=2] alpha = numpy.full( (size,states_num), 0, dtype=numpy.float64 ) #numpy.zeros( (size, states_num ))

        alpha[0,:] = logpi + logb[:, int(emissions[0]) ]
        for i in range(1,size):
            for s in range(states_num):

                alpha[i,s] = self.log_sum( alpha[i-1,:]+ loga[:,s] )

            #print(  numpy.exp(alpha[i,:]) )
            alpha[i,:] = alpha[i,:] + logb[:, int(emissions[i]) ]

        return alpha

    cpdef numpy.ndarray[float_t, ndim=2] backward(self, numpy.ndarray[int_t, ndim=1] emissions):
        """From emission sequence calculate the backward variables beta) given model parameters.
           Return logaritmus of probabilities.
        """
        cdef numpy.ndarray[float_t, ndim=2] loga = self._loga
        cdef numpy.ndarray[float_t, ndim=2] logb = self._logb
        cdef numpy.ndarray[float_t, ndim=1] logpi = self._logpi
        cdef int i, s, size, states_num

        size = emissions.shape[0]
        states_num = self._loga.shape[0]
        cdef numpy.ndarray[float_t, ndim=2] beta = numpy.full( (size,states_num), 0, dtype=numpy.float64 ) #numpy.zeros( (size, states_num ))

        beta[-1,:] = 0  #log(1) = 0
        for i in range(size-2, -1,-1):
            for s in range(states_num):
                beta[i,s] = self.log_sum( beta[i+1,:] + loga[s,:] + logb[:, int(emissions[i+1]) ] )

        return beta


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

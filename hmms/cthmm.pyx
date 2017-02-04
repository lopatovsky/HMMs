#import pyximport; pyximport.install()
from hmms import hmm
from hmms cimport hmm
import numpy
import scipy.linalg
cimport numpy
cimport cython
from cython cimport view

import random

#cython: wraparound=False
#cython: boundscheck=False
#cython: nonecheck=False

ctypedef numpy.float64_t float_t
ctypedef numpy.int_t int_t


cdef class CtHMM(hmm.HMM):

    cdef numpy.ndarray _q

    cdef dict tmap
    cdef float_t [:,:,:] _pt

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
        pt = numpy.empty( q.shape )

        b = numpy.exp( self._logb )
        pi = numpy.exp( self._logpi )

        states = numpy.empty(size,dtype=int)
        emissions = numpy.empty(size,dtype=int)
        times = numpy.empty(size,dtype=int)

        current_state = numpy.random.choice( pi.shape[0], 1, p= pi)
        current_time = 0;

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


    #TODO implement variant for square and multiplay
    cdef _prepare_matrices( self, numpy.ndarray[int_t, ndim=1] times ):
        """Will pre-count exponencials of matrices for all different time intervals"""
        cdef numpy.ndarray[float_t, ndim=2] q = self._q
        cdef float_t [:,:] pt = numpy.empty( (q.shape[0],q.shape[0]) , dtype=numpy.float64 )

        self._pt = numpy.empty( (times.shape[0],q.shape[0],q.shape[0]) , dtype=numpy.float64 ) #TODO may be uselessly too big
        self.tmap = {} # TODO isn't dict to slow?
        cdef int interval, cnt = 0

        for i in range ( 1, times.shape[0] ):
            #TODO double intervals
            interval = times[i] - times[i-1]
            if interval not in self.tmap:
               #print( "int: ", interval)
               #print(scipy.linalg.expm( q * interval ))

               pt = scipy.linalg.expm( q * interval )  #TODO copy directly in the 3D array
               self._pt[cnt,:,:] = pt

               self.tmap[ interval ] = cnt
               cnt+= 1

        #print("tmap")
        #print(self.tmap)


    #TODO pridaj ako test na prepare matrices
    def zmaz_ma( self, times ):
        self._prepare_matrices( times )
        for i in range ( 1, times.shape[0] ):
            interval = times[i] - times[i-1]
            print("i:",interval)
            print( numpy.asarray( self._pt[ self.tmap[ interval ] ]  ) )


    cpdef numpy.ndarray[float_t, ndim=2] forward(self, numpy.ndarray[int_t, ndim=1] times ,numpy.ndarray[int_t, ndim=1] emissions):
        """Method for the single call of forward algorithm"""
        self._prepare_matrices( times )
        return self._forward( times, emissions )


    cdef numpy.ndarray[float_t, ndim=2] _forward(self, numpy.ndarray[int_t, ndim=1] times ,numpy.ndarray[int_t, ndim=1] emissions):
        """From emission sequence calculate the forward variables (alpha) given model parameters.
           Return logaritmus of probabilities.
        """
        cdef numpy.ndarray[float_t, ndim=2] logb = self._logb
        cdef numpy.ndarray[float_t, ndim=1] logpi = self._logpi
        cdef int i, s, size, states_num, interval

        size = emissions.shape[0]
        states_num = self._q.shape[0]
        cdef numpy.ndarray[float_t, ndim=2] alpha = numpy.empty( (size,states_num), dtype=numpy.float64 )


        alpha[0,:] = logpi + logb[:, int(emissions[0]) ]
        for i in range(1,size):

            interval = times[i] - times[i-1]

            for s in range(states_num):

                alpha[i,s] = self.log_sum( alpha[i-1,:]
                                         + numpy.exp( numpy.asarray( self._pt[ self.tmap[ interval ],:,s]  ) ) ) #TODO probably can be optimised omitting exp

            #print(  numpy.exp(alpha[i,:]) )
            alpha[i,:] = alpha[i,:] + logb[:, int(emissions[i]) ]

        return alpha

    cpdef numpy.ndarray[float_t, ndim=2] backward(self, numpy.ndarray[int_t, ndim=1] times, numpy.ndarray[int_t, ndim=1] emissions):
        """From emission sequence calculate the backward variables beta) given model parameters.
           Return logaritmus of probabilities.
        """
        cdef numpy.ndarray[float_t, ndim=2] logb = self._logb
        cdef numpy.ndarray[float_t, ndim=1] logpi = self._logpi
        cdef int i, s, size, states_num, interval

        size = emissions.shape[0]
        states_num = self._q.shape[0]
        cdef numpy.ndarray[float_t, ndim=2] beta = numpy.empty( (size,states_num), dtype=numpy.float64 )

        beta[-1,:] = 0  #log(1) = 0
        for i in range(size-2, -1,-1):

            interval = times[i+1] - times[i]

            for s in range(states_num):
                beta[i,s] = self.log_sum( beta[i+1,:] + logb[:, int(emissions[i+1]) ]
                          + numpy.exp( numpy.asarray( self._pt[ self.tmap[ interval ],s,:]  ) ) )

        return beta

    cpdef float_t log_sum(self, numpy.ndarray[float_t, ndim=1] vec ):
        """Count sum of items in vec, that contain logaritmic probabilities using log-sum-exp trick"""
        cdef float_t max_p              # faster for:  max_p = numpy.amax( vec )
        cdef int i                      #
        max_p = vec[0]                  #
        for i in range(1,vec.shape[0]):   #
            if max_p < vec[i] : max_p = vec[i] #
        return max_p + numpy.log( numpy.sum( numpy.exp( vec - max_p ) ) )

    cpdef float_t log_sum_elem(self, float_t x, float_t y ):
        """Count sum of two items, that contain logaritmic probabilities using log-sum-exp trick"""
        cdef float_t max_p
        if x > y: max_p = x
        else    : max_p = y
        return max_p + numpy.log( numpy.exp( x - max_p ) + numpy.exp( y - max_p ) )



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

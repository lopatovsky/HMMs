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

    cpdef numpy.ndarray[float_t, ndim=2] backward(self, numpy.ndarray[int_t, ndim=1] times ,numpy.ndarray[int_t, ndim=1] emissions):
        """Method for the single call of backward algorithm"""
        self._prepare_matrices( times )
        return self._backward( times, emissions )


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

    cpdef numpy.ndarray[float_t, ndim=2] _backward(self, numpy.ndarray[int_t, ndim=1] times, numpy.ndarray[int_t, ndim=1] emissions):
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

    cpdef numpy.ndarray[float_t, ndim=2] single_state_prob( self, numpy.ndarray[float_t, ndim=2] alpha, numpy.ndarray[float_t, ndim=2] beta ):
        """Given forward and backward variables, count the probability for any state in any time"""
        cdef numpy.ndarray[float_t, ndim=2] gamma
        cdef float_t max_p, log_sum

        gamma = alpha + beta
        for i in range(gamma.shape[0]):
            gamma[i] -= self.log_sum(gamma[i])

        return gamma

    cpdef numpy.ndarray[float_t, ndim=3] double_state_prob( self, numpy.ndarray[float_t, ndim=2] alpha,
                                                                  numpy.ndarray[float_t, ndim=2] beta,
                                                                  numpy.ndarray[int_t, ndim=1  ] times,
                                                                  numpy.ndarray[int_t, ndim=1  ] emissions):
        """Given forward and backward variables, count the probability for transition from any state x to any state y in any time"""
        cdef numpy.ndarray[float_t, ndim=3] ksi = numpy.empty( (alpha.shape[0]-1,alpha.shape[1],alpha.shape[1]) , dtype=numpy.float64 )
        cdef numpy.ndarray[float_t, ndim=2] logb = self._logb
        cdef int interval

        for t in range( ksi.shape[0]):

            interval = times[t+1] - times[t]

            for i in range( ksi.shape[1]):
                for j in range( ksi.shape[2]):
                    ksi[t,i,j] = alpha[t,i]                                              \
                               + numpy.exp( self._pt[ self.tmap[ interval ],i,j] )       \
                               + logb[j, emissions[t+1] ] + beta[t+1,j]

            ksi[t,:,:] -= self.log_sum( ksi[t,:,:].flatten()  )

        #print(numpy.exp(ksi))

        return ksi  #Note: actually for use in Baum welch algorithm, it wouldn't need to store whole array.

    #TODO rename and change doc
    cpdef baum_welch(self, numpy.ndarray[int_t, ndim=2] times, numpy.ndarray[int_t, ndim=2] data, int iterations = 10 ):
        """Estimate parameters by Baum-Welch algorithm.
           Input array data is the numpy array of observation sequences.
        """

        self._prepare_matrices( times )

        cdef numpy.ndarray[float_t, ndim=1] gamma_sum, pi_sum, gamma_full_sum, gamma_part_sum
        cdef numpy.ndarray[float_t, ndim=2] alpha, beta, gamma, ksi_sum, obs_sum
        cdef numpy.ndarray[float_t, ndim=3] ksi

        #start_time = time.time()
        #...
        #print("--- %s seconds ---" % (time.time() - start_time))

        cdef int s_num = self._logb.shape[0]  #number of states
        cdef int o_num = self._logb.shape[1]  #number of possible observation symbols (emissions)

        for i in range( iterations ):

            #print("iter ", i)

            ksi_sum = numpy.full( ( s_num, s_num ) , numpy.log(0), dtype=numpy.float64 )
            obs_sum = numpy.full( ( s_num, o_num ) , numpy.log(0), dtype=numpy.float64 )  #numpy can samewhat handle infinities or at least exp(log(0)) = 0
            pi_sum  = numpy.full(  s_num , numpy.log(0), dtype=numpy.float64 )
            #gamma_part_sum  = numpy.full(  s_num , numpy.log(0), dtype=numpy.float64 )
            gamma_full_sum  = numpy.full(  s_num , numpy.log(0), dtype=numpy.float64 )
            gamma_sum = numpy.empty( s_num , dtype=numpy.float64 )


            for t , row in zip( times,data ):

                alpha = self.forward ( t, row )
                beta =  self.backward( t, row )

                gamma = self.single_state_prob( alpha, beta )
                ksi = self.double_state_prob( alpha, beta, t, row )



                #expected number of being in state i in time 0
                for i in range( s_num ):
                    pi_sum[i] = self.log_sum_elem( pi_sum[i], gamma[0,i] )


                """not yet used"""
                #expected number of transition from i to j
                for i in range( s_num ):
                    for j in range( s_num ):
                        ksi_sum[i,j] = self.log_sum_elem( ksi_sum[i,j], self.log_sum( ksi[:,i,j] ) )


                #expected number of visiting state i and observing symbol v
                for t in range( row.shape[0] ):
                    for i in range( s_num ):
                        obs_sum[i,row[t]] = self.log_sum_elem( obs_sum[i,row[t]], gamma[t,i] )

                #expected number of visiting state i
                for i in range( s_num ):
                    gamma_sum[i] = self.log_sum( gamma[:,i] )

                #sum gamma to the whole dataset array
                for i in range ( s_num ):
                    gamma_full_sum[i] = self.log_sum_elem( gamma_full_sum[i], gamma_sum[i] )

            #Update parameters:

            #initial probabilities estimation
            self._logpi = pi_sum - numpy.log( data.shape[0] )  #average
            #observetion symbol emission probabilities estimation
            self._logb = (obs_sum.T - gamma_full_sum).T

            #print( numpy.exp( self._logpi ) )
            #print( numpy.exp( self._loga ) )
            #print( numpy.exp( self._logb ) )






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

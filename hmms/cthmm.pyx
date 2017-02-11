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
    cdef float_t [:,:,:,:] _n_exp

    cdef numpy.ndarray _logb
    cdef numpy.ndarray _logpi
    cdef int time_n  #number of different time intervals

    @property
    def time_n(self):
        return self.time_n

    @property
    def q(self):
        return self._q

    @property
    def b(self):
        return numpy.exp( self._logb )

    @property
    def pi(self):
        return numpy.exp( self._logpi )

    @property
    def params( self ):
        return( self.q, self.b, self.pi )

    def __init__(self, Q,B,Pi):
        """Initialize the DtHMM by given parameters."""
        numpy.seterr( divide = 'ignore' )  #ignore warnings, when working with log(0) = -inf
        self.set_params( Q,B,Pi )

    @classmethod
    def from_file( cls, path ):
        """Initialize the class by reading parameters from file"""
        return cls( *CtHMM.get_params_from_file(path) )

    @staticmethod
    def get_params_from_file( path ):
        """Get parameters by reading them from .npz file"""
        npz = numpy.load( path )
        return ( npz['q'], npz['b'], npz['pi'] )

    def save_params( self, path ):
        """Save parameters in the file given by 'path'"""
        numpy.savez( path, q=self.q, b=self.b, pi=self.pi )

    def set_params( self, Q, B, Pi):
        """Set parameters as their logs to avoid underflow"""
        self._q = Q
        self._logb = numpy.log(B)
        self._logpi = numpy.log(Pi)

    def get_dthmm_params( self, time_step = 1 ):
        """Transform the jump rate matrix to transition matrix for discrete time of length time_step"""
        A = scipy.linalg.expm( self.q * time_step )  #the transition rate is set as the one time unit probabilities of continuos model
        B = self.b
        Pi = self.pi
        return (A,B,Pi)

    @classmethod
    def random( cls, s, o ):
        """Initialize the class by random parameters of 's' hidden states and 'o' output variables"""
        return cls( *CtHMM.get_random_params( s, o ) )

    def set_params_random( self, s, o ):
        """Set parameters by random. Size of 's' hidden states and 'o' output variables"""
        self.set_params( *CtHMM.get_random_params( s, o ) )

    def set_params_from_file( self, path ):
        """Set parameters by reading them from file"""
        self.set_params( *CtHMM.get_params_from_file(path) )


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

        for i in range( s ):
            b[i,:] = CtHMM.get_random_vector(o)
        pi = CtHMM.get_random_vector(s)

        return(q,b,pi)



    def generate(self, size, exp=0.5 ):
        """Randomly generate a sequence of states, times and emissions from model parameters."""
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
            time_interval = int( random.expovariate( exp ) ) + 1
            current_time += time_interval

            #print( time_interval )
            #print(q)
            #print( scipy.linalg.expm(q * time_interval ) )

            qt = scipy.linalg.expm(q * time_interval )

            current_state = numpy.random.choice( qt.shape[1],1, p = qt[ current_state,:].flatten() )

        return ( times, states, emissions )

    #TODO exp param
    def generate_data( self, size, **kargs ):
        """Generate multiple sequences of times and emissions from model parameters
           size = ( number of sequences, length of sequences  )
           **kargs:  states=True : return also sequence of states
        """
        e = numpy.empty( size, dtype=int )
        t = numpy.empty( size, dtype=int )
        s = numpy.empty( size, dtype=int )
        for i in range( size[0] ):
            t[i],s[i],e[i] = self.generate( size[1] )
        if ('states' in kargs) and kargs['states'] == True:
            return(t,s,e)

        return (t,e)


    #TODO implement variant for square and multiplay
    cdef _prepare_matrices_pt( self, numpy.ndarray[int_t, ndim=2] times ):
        """Will pre-count exponencials of matrices for all different time intervals"""

        cdef numpy.ndarray[float_t, ndim=2] q = self._q
        cdef float_t [:,:] pt = numpy.empty( (q.shape[0],q.shape[0]) , dtype=numpy.float64 )

        self._pt = numpy.empty( (times.shape[0]*times.shape[1],q.shape[0],q.shape[0]) , dtype=numpy.float64 ) #TODO may be uselessly too big
        self.tmap = {} # TODO isn't dict to slow?
        cdef int interval, cnt = 0

        for i in range( times.shape[0] ):
            for j in range ( 1, times.shape[1] ):
                #TODO double intervals
                interval = times[i,j] - times[i,j-1]
                if interval not in self.tmap:
                   #print( "int: ", interval)
                   #print(scipy.linalg.expm( q * interval ))
                   #print("int", interval)
                   #print("Q", q)

                   pt = scipy.linalg.expm( q * interval )  #TODO copy directly in the 3D array?
                   self._pt[cnt,:,:] = pt

                   self.tmap[ interval ] = cnt
                   cnt+= 1

        self.time_n = cnt

        #print("tmap")
        #print(self.tmap)

    cdef _prepare_matrices_n_exp( self ):
        """Will pre-count integrals $\int_0^t( e^{Qx}_{k,i} e^{Q(t-x)}_{j,l} dx$ for any states $i,j,k,l \in$ hidden states """

        cdef int s_num = self._q.shape[0]

        #Construct auxiliary matrix A = [[Q,B][0,Q]]

        cdef numpy.ndarray[float_t, ndim=2] A = numpy.zeros(  ( 2*s_num, 2*s_num ), dtype=numpy.float64  )

        A[:s_num,:s_num] = self._q
        A[s_num:,s_num:] = self._q

        cdef i,j
        cdef float_t [:,:] temp = numpy.empty( (2*s_num,2*s_num) , dtype=numpy.float64 )

        self._n_exp = numpy.empty( (s_num, s_num, 2*s_num, 2*s_num) , dtype=numpy.float64 )

        for i in range(s_num):
            for j in range( s_num ):
                A[i,s_num + j] = 1  # set the subpart matrix B

                temp = scipy.linalg.expm( A )  #TODO copy directly in the 4D array

                self._n_exp[i,j,:,:] = temp

                A[i,s_num + j] = 0  # zero the subpart matrix B

    cpdef numpy.ndarray[float_t, ndim=2] forward(self, numpy.ndarray[int_t, ndim=1] times ,numpy.ndarray[int_t, ndim=1] emissions):
        """Method for the single call of forward algorithm"""
        self._prepare_matrices_pt( numpy.array( [times] ) )
        return self._forward( times, emissions )

    cpdef numpy.ndarray[float_t, ndim=2] backward(self, numpy.ndarray[int_t, ndim=1] times ,numpy.ndarray[int_t, ndim=1] emissions):
        """Method for the single call of backward algorithm"""
        self._prepare_matrices_pt( numpy.array( [times] ) )
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


        #print("FORWARD")
        #print("should be 0:",self.tmap[ 1 ])
        #print( numpy.asarray( self._pt[ self.tmap[ 1 ],:,:]  ) )
        #print("FORWARD")

        alpha[0,:] = logpi + logb[:, int(emissions[0]) ]
        for i in range(1,size):

            interval = times[i] - times[i-1]
            for s in range(states_num):

                alpha[i,s] = self.log_sum( alpha[i-1,:]
                                         + numpy.log( self._pt[ self.tmap[ interval ],:,s]  ) ) #TODO probably can be optimised omitting exp

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
                          + numpy.log(  self._pt[ self.tmap[ interval ],s,:]   ) )

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
                               + numpy.log( self._pt[ self.tmap[ interval ],i,j] )       \
                               + logb[j, emissions[t+1] ] + beta[t+1,j]

            ksi[t,:,:] -= self.log_sum( ksi[t,:,:].flatten()  )

        #print(numpy.exp(ksi))

        return ksi  #Note: actually for use in Baum welch algorithm, it wouldn't need to store whole array.

    cpdef emission_estimate(self, numpy.ndarray[int_t, ndim=1] times, numpy.ndarray[int_t, ndim=1] emissions ):
        """From given emission sequence calculate the likelihood estimation given model parameters"""
        return  self.log_sum( self.forward( times,emissions )[-1,:] )

    cpdef data_estimate( self, numpy.ndarray[int_t, ndim=2] times ,numpy.ndarray[int_t, ndim=2] data ):
        """From the set of given emission sequences in the data calculate their likelihood estimation given model parameters"""
        cdef float sm = 0
        for t,row in zip( times, data):
            sm += self.emission_estimate( t,row )
        return sm

    #TODO move to the small artificial class in art.py
    def baum_welch_graph( self, times, data, iteration =10  ):
        """Slower method for Baum-Welch that in evey algorithm iteration count the data estimation, so it could return its learning curve"""
        graph = numpy.empty(iteration+1)
        graph[0] = self.data_estimate( times, data)

        for i in range(1,iteration+1):
            self._baum_welch( times, data, 0,1 )
            graph[i] = self.data_estimate(times, data)

        return graph

    def baum_welch_graph2( self, times, data, iteration =10  ):
        """Slower method for Baum-Welch that in evey algorithm iteration count the data estimation, so it could return its learning curve"""
        graph = numpy.empty(iteration+1)
        graph[0] = self.data_estimate( times, data)

        for i in range(1,iteration+1):
            self._baum_welch( times, data, 1,1 )
            graph[i] = self.data_estimate(times, data)

        return graph

    def baum_welch( self, times, data, iteration = 10, **kvargs ):

        if 'method' in kvargs:
            if kvargs['method'] == 'lr':
                self._baum_welch( times, data, 1, iteration )
                return


        self._baum_welch( times, data, 0, iteration )



    #TODO rename and change doc
    cdef _baum_welch(self, numpy.ndarray[int_t, ndim=2] times, numpy.ndarray[int_t, ndim=2] data, int method , int iterations):
        """Estimate parameters by Baum-Welch algorithm.
           Input array data is the numpy array of observation sequences.
        """

        cdef numpy.ndarray[float_t, ndim=1] gamma_sum, pi_sum, gamma_full_sum, gamma_part_sum, tau
        cdef numpy.ndarray[int_t, ndim=1] t, row
        cdef numpy.ndarray[float_t, ndim=2] alpha, beta, gamma, obs_sum, eta, tA, temp
        cdef numpy.ndarray[float_t, ndim=3] ksi, ksi_sum
        cdef int i,j,k,l,tm,map_time,ix

        #start_time = time.time()
        #...
        #print("--- %s seconds ---" % (time.time() - start_time))

        cdef int s_num = self._logb.shape[0]  #number of states
        cdef int o_num = self._logb.shape[1]  #number of possible observation symbols (emissions)

        for i in range( iterations ):

            print("iter ", i)

            self._prepare_matrices_pt( times )

            ksi_sum = numpy.full( ( self.time_n, s_num, s_num ) , numpy.log(0), dtype=numpy.float64 )
            obs_sum = numpy.full( ( s_num, o_num ) , numpy.log(0), dtype=numpy.float64 )  #numpy can samewhat handle infinities or at least exp(log(0)) = 0
            pi_sum  = numpy.full(  s_num , numpy.log(0), dtype=numpy.float64 )
            #gamma_part_sum  = numpy.full(  s_num , numpy.log(0), dtype=numpy.float64 )
            gamma_full_sum  = numpy.full(  s_num , numpy.log(0), dtype=numpy.float64 )
            gamma_sum = numpy.empty( s_num , dtype=numpy.float64 )



            for t , row in zip( times,data ):

                alpha = self._forward ( t, row )
                beta =  self._backward( t, row )

                gamma = self.single_state_prob( alpha, beta )
                ksi = self.double_state_prob( alpha, beta, t, row )
                #TODO sum probs for same delta(t).

                #print("ksi",ksi)


                #expected number of being in state i in time 0
                for i in range( s_num ):
                    pi_sum[i] = self.log_sum_elem( pi_sum[i], gamma[0,i] )

                #sum the ksi with same time interval together
                for tm in range( t.shape[0] - 1 ):

                    interval = t[tm+1]-t[tm]
                    map_time = self.tmap[ interval ]

                    #print("interval", interval)

                    for i in range(s_num):
                        for j in range( s_num ):

                            #print("ks_sum",ksi_sum[map_time,i,j], ksi[tm,i,j])
                            ksi_sum[map_time,i,j] = self.log_sum_elem( ksi_sum[map_time,i,j], ksi[tm,i,j] )
                            #print("ks_sum",ksi_sum[map_time,i,j], ksi[tm,i,j])

                #print("ksisum",ksi_sum)

                #print("C ksi", ksi_sum[0] )

                #expected number of visiting state i and observing symbol v
                for tm in range( row.shape[0] ):
                    for i in range( s_num ):
                        obs_sum[i,row[tm]] = self.log_sum_elem( obs_sum[i,row[tm]], gamma[tm,i] )

                #expected number of visiting state i
                for i in range( s_num ):
                    gamma_sum[i] = self.log_sum( gamma[:,i] )

                #sum gamma to the whole dataset array
                for i in range ( s_num ):
                    gamma_full_sum[i] = self.log_sum_elem( gamma_full_sum[i], gamma_sum[i] )


            #tau, eta = self.end_state_expectations( ksi_sum ) #TODO move all of these in separate function?
            tau = numpy.zeros( (s_num), dtype=numpy.float64 )
            eta = numpy.zeros( (s_num,s_num), dtype=numpy.float64 )

            tA = numpy.zeros( (s_num,s_num), dtype=numpy.float64 )

            temp = numpy.empty( (s_num*2,s_num*2), dtype=numpy.float64 )

            self._prepare_matrices_n_exp()

            print(numpy.asarray(self._n_exp[0,0]))

            #print("EXP 0,2 - 1-1")
            #print( numpy.asarray( self._n_exp[0,2] ) )
            #print( numpy.asarray( self._n_exp[1,1] ) )
            #print("EXP")

            for tm, ix in self.tmap.items():  #iterate trought all different time intervals

                #print("tm ix", tm, ix)

                for i in range(s_num):
                    for j in range( s_num ):

#                        if i==0 and j==0:
#                            print(i,j, numpy.asarray(self._n_exp[i,j]))
#                            print("t1",temp)

                        ##doesn't work - > temp = numpy.asarray(self._n_exp[i,j,:,:])
                        for k in range(s_num*2):
                            for l in range(s_num*2):
                                temp[k,l] = self._n_exp[i,j,k,l]


                        #temp = numpy.full( (s_num*2,s_num*2), 4.25 )

                        #TODO -numpy bug? temp as the ndarray is the same memory as the output tA array

                        tA  = numpy.linalg.matrix_power( temp , tm )[:s_num,s_num:]  #TODO cashing can save some O(2/3) of computations

#                        if i==0 and j==0:
#                            #tA[0,0] = 5
#                            print(i,j, numpy.asarray(self._n_exp[i,j]))
#                            print("d",temp)
#                            print("dA",tA)


                        #print("ta",tA)
                        #print("ksi", ksi_sum[ix] )
                        #print("log", numpy.log( tA ) )
                        #print("pt",numpy.asarray(self._pt[ ix ]))



                        if method == 0 :

                            if i == j:

                                tA /= self._pt[ ix ]

                                tau[i]  += numpy.exp( self.log_sum( (ksi_sum[ix] + numpy.log( tA ) ).flatten() ) )   #tau is not in log prob anymore.

                                #print(tau[i])

                            else:
                                tA *= self._q[i,j]
                                tA /= self._pt[ ix ]
                                eta[i,j] += numpy.exp( self.log_sum( (ksi_sum[ix] + numpy.log( tA ) ).flatten() ) )  #eta is not in log prob anymore.

                        if method == 1:

                            if i == j:

                                tA /= self._pt[ ix ]

                                tau[i]  += numpy.exp( self.log_sum( (ksi_sum[ix] + numpy.log( tA ) ).flatten() ) - numpy.log(tm) )   #tau is not in log prob anymore.

                                #print(tau[i])

                            else:
                                tA *= self._q[i,j]
                                tA /= self._pt[ ix ]
                                eta[i,j] += numpy.exp( self.log_sum( (ksi_sum[ix] + numpy.log( tA ) ).flatten() ) - numpy.log(tm) )  #eta is not in log prob anymore.

                print("tau\n",tau)


            print("tau\n",tau)
            #print("eta\n",eta)

            #Update parameters:
            if method == 0 or method == 1:
                #initial probabilities estimation
                self._logpi = pi_sum - numpy.log( data.shape[0] )  #average
                #observetion symbol emission probabilities estimation
                self._logb = (obs_sum.T - gamma_full_sum).T
                #jump rates matrice
                self._q = ( eta.T / tau ).T
                #print(  self._q  )
                for i in range( s_num ):
                    self._q[i,i] = - numpy.sum( self._q[i,:] )

                #print( numpy.exp( self._logpi ) )
                #print( self._q )
                #print( "CT-HMM qt, t = 1s  like A" )
                #print( scipy.linalg.expm( self._q ) )
                #print( numpy.exp( self._logb ) )
#
#            if method == 1:
#
#                lr = 0.5
#
#                #initial probabilities estimation
#                self._logpi = lr + self._logpi ,  (1.0-lr) +(  pi_sum - numpy.log( data.shape[0] )  ) #average
#                #observetion symbol emission probabilities estimation
#                self._logb = lr + self._logpi ,  (1.0-lr) +(  (obs_sum.T - gamma_full_sum).T )
#                #jump rates matrice
#                self._q = lr * self._logpi +  (1.0-lr) *(  ( eta.T / tau ).T )
#                #print(  self._q  )
#                for i in range( s_num ):
#                    self._q[i,i] = - numpy.sum( self._q[i,:] )




    #cdef ( numpy.ndarray[float_t, ndim=1], numpy.ndarray[float_t, ndim=2] ) end_state_expectations( self, numpy.ndarray[float_t, ndim=3] ksi_sum ):
        #self._prepare_matrices_n_exp()

    cpdef float_t log_sum(self, numpy.ndarray[float_t, ndim=1] vec ):
        """Count sum of items in vec, that contain logaritmic probabilities using log-sum-exp trick"""
        cdef float_t max_p              # faster for:  max_p = numpy.amax( vec )
        cdef int i                      #
        max_p = vec[0]                  #
        for i in range(1,vec.shape[0]):   #
            if max_p < vec[i] : max_p = vec[i] #

        if numpy.isinf( max_p ): return max_p  #to avoid nan in (inf-inf)

        return max_p + numpy.log( numpy.sum( numpy.exp( vec - max_p ) ) )

    cpdef float_t log_sum_elem(self, float_t x, float_t y ):
        """Count sum of two items, that contain logaritmic probabilities using log-sum-exp trick"""
        cdef float_t max_p
        if x > y: max_p = x
        else    : max_p = y

        if numpy.isinf( max_p ): return max_p  #to avoid nan in (inf-inf)

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

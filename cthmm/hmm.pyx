import numpy
import random
import time  #performance meassure
cimport numpy
cimport cython

#cython: wraparound=False
#cython: boundscheck=False
#cython: nonecheck=False

ctypedef numpy.float64_t float_t
ctypedef numpy.int_t int_t

cdef class HMM:

    """Parameters _loga, _logb, _logpi are log likelihoods of _a, _b and _pi used to avoid underflow."""
    cdef numpy.ndarray _a #todo used only in generate, maybe can be erased.
    cdef numpy.ndarray _b #todo parameters without log will not be recalculated after bw alg. delete them.
    cdef numpy.ndarray _pi #
    cdef numpy.ndarray _loga
    cdef numpy.ndarray _logb
    cdef numpy.ndarray _logpi

    def __init__(self, A, B, Pi):
        """Initialize the HMM by given parameters."""
        self._a = A
        self._b = B
        self._pi = Pi
        self._loga = numpy.log(A)
        self._logb = numpy.log(B)
        self._logpi = numpy.log(Pi)

    def check_parameters(self):
        pass #TODO

    def generate(self, size ):
        """Randomly generate a sequence of states and emissions from model parameters."""
        states = numpy.empty(size,dtype=int)
        emissions = numpy.empty(size,dtype=int)
        current_state = numpy.random.choice( self._pi.shape[0], 1, p= self._pi)
        for i in range(size):
            states[i] = current_state
            emissions[i] =  numpy.random.choice( self._b.shape[1],1, p = self._b[ current_state,:].flatten() )
            current_state = numpy.random.choice( self._a.shape[1],1, p = self._a[ current_state,:].flatten() )
        return ( states, emissions )

    #cpdef estimate(self, states, emissions):
    #    """From given state and emission sequence calculate their likelihood estimation given model parameters"""
    #

    cpdef emission_estimate(self, numpy.ndarray[int_t, ndim=1] emissions ):
        """From given emission sequence calculate the likelihood estimation given model parameters"""
        return  self.log_sum( self.forward( emissions )[-1,:] )

    cpdef numpy.ndarray[float_t, ndim=2] forward(self, numpy.ndarray[int_t, ndim=1] emissions):
        """From emission sequence calculate the forward variables (alpha) given model parameters.
           Return logaritmus of probabilities.
        """
        cdef numpy.ndarray[float_t, ndim=2] loga = self._loga
        cdef numpy.ndarray[float_t, ndim=2] logb = self._logb
        cdef numpy.ndarray[float_t, ndim=1] logpi = self._logpi
        cdef int i, s, size, states_num,

        size = emissions.shape[0]
        states_num = self._a.shape[0]
        cdef numpy.ndarray[float_t, ndim=2] alpha = numpy.full( (size,states_num), 0, dtype=numpy.float64 ) #numpy.zeros( (size, states_num ))

        alpha[0,:] = logpi + logb[:, int(emissions[0]) ]
        for i in range(1,size):
            for s in range(states_num):

                alpha[i,s] = self.log_sum( alpha[i-1,:]+loga[:,s] )

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
        states_num = self._a.shape[0]
        cdef numpy.ndarray[float_t, ndim=2] beta = numpy.full( (size,states_num), 0, dtype=numpy.float64 ) #numpy.zeros( (size, states_num ))

        beta[-1,:] = 0  #log(1) = 0
        for i in range(size-2, -1,-1):
            for s in range(states_num):
                beta[i,s] = self.log_sum( beta[i+1,:] + loga[s,:] + logb[:, int(emissions[i+1]) ] )

        return beta

    cpdef viterbi(self, numpy.ndarray[int_t, ndim=1] emissions):
        """From given emission sequence and parameters calculate the most likely state sequence"""

        cdef numpy.ndarray[float_t, ndim=2] loga = self._loga
        cdef numpy.ndarray[float_t, ndim=2] logb = self._logb
        cdef numpy.ndarray[float_t, ndim=1] logpi = self._logpi
        cdef int i, s, size, states_num,
        cdef float_t max_p

        size = emissions.shape[0]
        states_num = self._a.shape[0]
        cdef numpy.ndarray[float_t, ndim=2] delta = numpy.full( (size,states_num), 0, dtype=numpy.float64 ) #numpy.zeros( (size, states_num ))
        cdef numpy.ndarray[int_t, ndim=2] psi = numpy.full( (size,states_num), 0, dtype=numpy.int ) #numpy.zeros( (size, states_num ))

        delta[0,:] = logpi + logb[:, int(emissions[0]) ]
        psi[0,:] = 0
        for i in range(1,size):
            for s in range(states_num):

                delta[i,s] = delta[i-1,0] + loga[0,s]
                psi[i,s] = 0

                for r in range(1,states_num):
                    if delta[i,s] < delta[i-1,r] + loga[r,s]:
                        delta[i,s] = delta[i-1,r] + loga[r,s]
                        psi[i,s] = r

                delta[i,s] += logb[s,emissions[i]]
        #print(numpy.exp(delta))
        #print(psi)

        max_p = delta[-1,0]
        p = 0

        for s in range(1,states_num):
            if max_p < delta[-1,s]:
                max_p = delta[-1,s]
                p = s

        cdef numpy.ndarray[int_t, ndim=1] path = numpy.full( size, 0, dtype=numpy.int )

        for i in range(size-1,-1,-1):
            path[i] = p
            p = psi[i,p]

        return ( max_p, path )

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
                                                                  numpy.ndarray[int_t, ndim=1  ] emissions):
        """Given forward and backward variables, count the probability for transition from any state x to any state y in any time"""
        cdef numpy.ndarray[float_t, ndim=3] ksi = numpy.empty( (alpha.shape[0]-1,alpha.shape[1],alpha.shape[1]) , dtype=numpy.float64 )
        cdef numpy.ndarray[float_t, ndim=2] loga = self._loga  #Such declaration make it cca 10% faster
        cdef numpy.ndarray[float_t, ndim=2] logb = self._logb


        for t in range( ksi.shape[0]):
            for i in range( ksi.shape[1]):
                for j in range( ksi.shape[2]):
                    ksi[t,i,j] = alpha[t,i] + loga[i,j] + logb[j, emissions[t+1] ] + beta[t+1,j]
            ksi[t,:,:] -= self.log_sum( ksi[t,:,:].flatten()  )

        return ksi  #Note: actually for use in Baum welch algorithm, it wouldn't need to store whole array.


    #TODO - a bit useless restriction on 2d matrix of data, if they do not need to have some length at all.
    cpdef baum_welch(self, numpy.ndarray[int_t, ndim=2] data):
        """Estimate parameters by Baum-Welch algorithm.
           Input array data is the numpy array of observation sequences.
        """
        cdef numpy.ndarray[float_t, ndim=1] gamma_sum, pi_sum, gamma_full_sum, gamma_part_sum
        cdef numpy.ndarray[float_t, ndim=2] alpha, beta, gamma, ksi_sum, obs_sum
        cdef numpy.ndarray[float_t, ndim=3] ksi

        #start_time = time.time()
        #...
        #print("--- %s seconds ---" % (time.time() - start_time))

        cdef int s_num = self._logb.shape[0]  #number of states
        cdef int o_num = self._logb.shape[1]  #number of possible observation symbols (emissions)


        for i in range(2):

            print("iter ", i)

            ksi_sum = numpy.full( ( s_num, s_num ) , numpy.log(0), dtype=numpy.float64 )
            obs_sum = numpy.full( ( s_num, o_num ) , numpy.log(0), dtype=numpy.float64 )  #numpy can samewhat handle infinities or at least exp(log(0)) = 0
            pi_sum  = numpy.full(  s_num , numpy.log(0), dtype=numpy.float64 )
            gamma_part_sum  = numpy.full(  s_num , numpy.log(0), dtype=numpy.float64 )
            gamma_full_sum  = numpy.full(  s_num , numpy.log(0), dtype=numpy.float64 )



            for row in data:

                alpha = self.forward ( row )
                beta =  self.backward( row )

                gamma = self.single_state_prob( alpha, beta )
                ksi = self.double_state_prob( alpha, beta, row )

                gamma_sum = numpy.empty( s_num , dtype=numpy.float64 )

                #expected number of being in state i in time 0
                for i in range( s_num ):
                    pi_sum[i] = self.log_sum_elem( pi_sum[i], gamma[0,i] )

                #print("pi")
                #print( numpy.exp(gamma[0,:]) )
                #print( numpy.exp( pi_sum ) )

                #expected number of transition from i to j
                for i in range( s_num ):
                    for j in range( s_num ):
                        ksi_sum[i,j] = self.log_sum_elem( ksi_sum[i,j], self.log_sum( ksi[:,i,j] ) )

                #expected number of transition from state i
                for i in range( s_num ):
                    gamma_sum[i] = self.log_sum( gamma[:-1,i] )

                #sum gamma to the whole dataset array
                for i in range ( s_num ):
                    gamma_part_sum[i] = self.log_sum_elem( gamma_part_sum[i], gamma_sum[i] )

                #expected number of visiting state i and observing symbol v
                for t in range( row.shape[0] ):
                    for i in range( s_num ):
                        obs_sum[i,row[t]] = self.log_sum_elem( obs_sum[i,row[t]], gamma[t,i] )

                #expected number of visiting state i
                for i in range( s_num ):  #full length sum
                    gamma_sum[i] = self.log_sum_elem( gamma_sum[i], gamma[-1,i]  )

                #sum gamma to the whole dataset array
                for i in range ( s_num ):
                    gamma_full_sum[i] = self.log_sum_elem( gamma_full_sum[i], gamma_sum[i] )

            #Update parameters:

            #initial probabilities estimation
            self._logpi = pi_sum - numpy.log( data.shape[0] )  #average
            #transition matrix estimation
            self._loga = (ksi_sum.T - gamma_part_sum).T
            #observetion symbol emission probabilities estimation
            self._logb = (obs_sum.T - gamma_full_sum).T

            print( numpy.exp( self._logpi ) )
            print( numpy.exp( self._loga ) )
            print( numpy.exp( self._logb ) )




    def from_file( self,file_path ):
        """Initialize the HMM by the file from the file_path storing the parameters A,B,Pi""" ##TODO define the file format.
        print("hello file")

    def meow(self):
        """Make the HMM to meow"""
        print('meow!')



def get_random_vector( s ):
    """Generate random vector of size (s), with all values summing to one"""
    vec = numpy.random.random(s)
    return vec / numpy.sum(vec)


def get_random_parameters( s, o ):
    """Generate random parameters A,B and Pi, for number of hidden states (s) and output variables (o)"""

    a = numpy.empty( [s,s] )
    b = numpy.empty( [s,o] )
    pi = numpy.empty( s )

    for i in range( a.shape[0] ):
        a[i,:] = get_random_vector(s)
    for i in range( b.shape[0]):
        b[i,:] = get_random_vector(o)
    pi = get_random_vector(s)

    return (a,b,pi)



def bw_test():

    print("--bw_test--")

    print("small test:")

    A = numpy.array([[0.9,0.1],[0.4,0.6]])
    B = numpy.array([[0.9,0.1],[0.2,0.8]])
    pi = numpy.array( [0.8,0.2] )
    hmm = HMM(A,B,pi)

    data = numpy.array([[0,1,1]]);
    hmm.baum_welch( data )

    print( hmm.viterbi( data[0] ) )

    print("big test")

    print( get_random_vector( 5 ) )

    A = numpy.array([[0.9,0.07,0.03],[0.2,0.6,0.2],[0.15,0.05,0.8]])
    B = numpy.array([[0.9,0.005,0.07,0.005,0.02],[0.3,0.2,0.4,0.04,0.06],[0.5,0.03,0.03,0.14,0.3] ])
    pi = numpy.array( [0.8,0.15,0.05] )
    hmm = HMM(A,B,pi)


    num = 50
    data_len = 50
    s = numpy.empty( (num, data_len), dtype=int )
    e = numpy.empty( (num, data_len), dtype=int )

    for i in range(num):
        s[i], e[i] = hmm.generate( data_len )
        print( e[i] )

    hmmr = HMM( A,B,pi )
    #hmmr = HMM( *get_random_parameters(3,5) )
    hmmr.baum_welch( e )


    #print(e)
    real_est = 0
    guess_est = 0
    real_vit = 0
    guess_vit = 0


    for i in range( e.shape[0] ):
        #print(e[i])

        real_est += numpy.exp( hmm.emission_estimate( e[i] ) )
        guess_est += numpy.exp( hmmr.emission_estimate( e[i] ) )

        va, vb = hmm.viterbi(e[i])
        var, vbr = hmmr.viterbi(e[i])
        real_vit += numpy.exp(va)
        guess_vit += numpy.exp(var)


    print("real_est", real_est)
    print("guess_est",guess_est)
    print("real_vit",real_vit)
    print("guess_vit",guess_vit)

    print("actual arrays: ")
    print(pi)
    print(A)
    print(B)



def main():
    #my_hmm = HMM()
    A = numpy.array([[0.9,0.1],[0.4,0.6]])
    B = numpy.array([[0.9,0.1],[0.2,0.8]])
    pi = numpy.array( [0.8,0.2] )
    hmm = HMM(A,B,pi)
    (s,e) = hmm.generate(3)
    print(s)
    print(e)
    e = numpy.array([0,1])

    print( numpy.exp(hmm.forward(e) ) )
    print( numpy.exp(hmm.backward(e) ) )
    print( (hmm.emission_estimate(e) ) )

    print("Viterbi: ")

    ob = numpy.array([0,0,0,1,1,0,1,1,0,0,0,0,0,1,0,0,1,1,0,0,0,1,0,1,1,1,1,0,0,1,1,1])
    t1 = numpy.array([0,1,0,1,1])

    p, path = hmm.viterbi( ob )
    print( numpy.exp(p) )
    print(ob)
    print( path )

    return

    bw_test()


    #hmm2 = HMM.from_parameters(A,B,pi)
    #hmm2 = HMM.from_file("x.hmm")
    #my_hmm.meow()

if __name__ == "__main__":
    main()

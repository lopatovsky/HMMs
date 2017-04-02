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

cdef class DtHMM:

    """Parameters _loga, _logb, _logpi are log likelihoods to avoid underflow."""
    cdef numpy.ndarray _loga
    cdef numpy.ndarray _logb
    cdef numpy.ndarray _logpi


    @property
    def a(self):
        return numpy.exp( self._loga )

    @property
    def b(self):
        return numpy.exp( self._logb )

    @property
    def pi(self):
        return numpy.exp( self._logpi )

    @property
    def params( self ):
        return( self.a, self.b, self.pi )

    def __init__(self, A,B,Pi):
        """Initialize the DtHMM by given parameters."""
        numpy.seterr( divide = 'ignore' )  #ignore warnings, when working with log(0) = -inf
        self.set_params( A,B,Pi )

    @classmethod
    def from_file( cls, path ):
        """Initialize the class by reading parameters from file"""
        return cls( *DtHMM.get_params_from_file(path) )

    @classmethod
    def random( cls, s, o ):
        """Initialize the class by random parameters of 's' hidden states and 'o' output variables"""
        return cls( *DtHMM.get_random_params( s, o ) )

    def set_params( self, A, B, Pi):
        """Set parameters as their logs to avoid underflow"""
        self._loga = numpy.log(A)
        self._logb = numpy.log(B)
        self._logpi = numpy.log(Pi)

    def set_params_from_file( self, path ):
        """Set parameters by reading them from file"""
        self.set_params( *DtHMM.get_params_from_file(path) )

    def set_params_random( self, s, o ):
        """Set parameters by random. Size of 's' hidden states and 'o' output variables"""
        self.set_params( *DtHMM.get_random_params( s, o ) )

    def save_params( self, path ):
        """Save parameters in the file given by 'path'"""
        numpy.savez( path, a=self.a, b=self.b, pi=self.pi )


    @staticmethod
    def get_params_from_file( path ):
        """Get parameters by reading them from .npz file"""
        npz = numpy.load( path )
        return ( npz['a'], npz['b'], npz['pi'] )

    @staticmethod
    def get_random_vector( s ):
        """Generate random vector of size (s), with all values summing to one"""
        vec = numpy.random.random(s)
        return vec / numpy.sum(vec)

    @staticmethod
    def get_random_params( s, o ):
        """Generate random parameters A,B and Pi, for number of hidden states (s) and output variables (o)"""

        a = numpy.empty( [s,s] )
        b = numpy.empty( [s,o] )
        pi = numpy.empty( s )

        for i in range( a.shape[0] ):
            a[i,:] = DtHMM.get_random_vector(s)
        for i in range( b.shape[0]):
            b[i,:] = DtHMM.get_random_vector(o)
        pi = DtHMM.get_random_vector(s)

        return(a,b,pi)

    def check_parameters(self):
        pass #TODO


    #TODO test it
    def generate(self, size ):
        """Randomly generate a sequence of states and emissions from model parameters."""
        a = numpy.exp( self._loga )
        b = numpy.exp( self._logb )
        pi = numpy.exp( self._logpi )

        states = numpy.empty(size,dtype=int)
        emissions = numpy.empty(size,dtype=int)
        current_state = numpy.random.choice( pi.shape[0], 1, p= pi)
        for i in range(size):
            states[i] = current_state
            emissions[i] =  numpy.random.choice( b.shape[1],1, p = b[ current_state,:].flatten() )
            current_state = numpy.random.choice( a.shape[1],1, p = a[ current_state,:].flatten() )
        return ( states, emissions )

    def generate_data(self, size, **kargs ):
        """Generate multiple sequences of states and emissions from model parameters
           size = ( number of sequences, length of sequences  )
           **kargs:  times=True : return also equidistant sequence of times
        """
        e = numpy.empty( size, dtype=int )
        t = numpy.empty( size, dtype=int )
        s = numpy.empty( size, dtype=int )
        for i in range( size[0] ):
            s[i],e[i] = self.generate( size[1] )
            t[i] = numpy.arange(  size[1] )

        if ('times' in kargs) and kargs['times'] == True:
            return(t,s,e)

        return (s,e)

    #cpdef estimate(self, states, emissions):
    #    """From given state and emission sequence calculate their likelihood estimation given model parameters"""
    #

    cpdef emission_estimate(self, numpy.ndarray[int_t, ndim=1] emissions ):
        """From given emission sequence calculate the likelihood estimation given model parameters"""
        return  self.log_sum( self.forward( emissions )[-1,:] )

    cpdef data_estimate( self, numpy.ndarray[int_t, ndim=2] data ):
        """From the set of given emission sequences in the data calculate their likelihood estimation given model parameters"""
        cdef float sm = 0
        for row in data:
            sm += self.emission_estimate( row )
        return sm


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
        cdef numpy.ndarray[float_t, ndim=2] alpha = numpy.empty( (size,states_num), dtype=numpy.float64 ) #numpy.zeros( (size, states_num ))

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
        cdef numpy.ndarray[float_t, ndim=2] beta = numpy.empty( (size,states_num), dtype=numpy.float64 ) #numpy.zeros( (size, states_num ))

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
        states_num = self._loga.shape[0]
        cdef numpy.ndarray[float_t, ndim=2] delta = numpy.empty( (size,states_num), dtype=numpy.float64 ) #numpy.zeros( (size, states_num ))
        cdef numpy.ndarray[int_t, ndim=2] psi = numpy.empty( (size,states_num), dtype=numpy.int ) #numpy.zeros( (size, states_num ))

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

        if numpy.isinf( max_p ): return max_p  #to avoid nan in (inf-inf)

        return max_p + numpy.log( numpy.sum( numpy.exp( vec - max_p ) ) )

    cpdef float_t log_sum_elem(self, float_t x, float_t y ):
        """Count sum of two items, that contain logaritmic probabilities using log-sum-exp trick"""
        cdef float_t max_p
        if x > y: max_p = x
        else    : max_p = y

        if numpy.isinf( max_p ): return max_p  #to avoid nan in (inf-inf)

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

        #print(numpy.exp(ksi))

        return ksi  #Note: actually for use in Baum welch algorithm, it wouldn't need to store whole array.

    #TODO - merge the estimation counting with the forward alg and baum welch procedure.
    #TODO move to the small artificial class in art.py - nope
    def baum_welch_graph( self, data, iteration =10 ):
        """Slower method for Baum-Welch that in evey algorithm iteration count the data estimation, so it could return its learning curve"""
        graph = numpy.empty(iteration+1)
        graph[0] = self.data_estimate(data)

        for i in range(1,iteration+1):
            self.baum_welch( data, 1 )
            graph[i] = self.data_estimate(data)

        return graph


    #TODO - a bit useless restriction on 2d matrix of data, if they do not need to have some length at all.
    #TODO2 - change default value to -1 - convergence
    #TODO3 - examine if warning  can cause some problems "/home/jamaisvu/Desktop/CT-DtHMM/tests/test_hmm.py:160: RuntimeWarning: divide by zero encountered in log"


    cpdef baum_welch(self, numpy.ndarray[int_t, ndim=2] data, int iterations = 10 ):
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
        cdef int i,j,t


        for i in range( iterations ):



            #print("iter ", i)

            ksi_sum = numpy.full( ( s_num, s_num ) , numpy.log(0), dtype=numpy.float64 )
            obs_sum = numpy.full( ( s_num, o_num ) , numpy.log(0), dtype=numpy.float64 )  #numpy can samewhat handle infinities or at least exp(log(0)) = 0
            pi_sum  = numpy.full(  s_num , numpy.log(0), dtype=numpy.float64 )
            gamma_part_sum  = numpy.full(  s_num , numpy.log(0), dtype=numpy.float64 )
            gamma_full_sum  = numpy.full(  s_num , numpy.log(0), dtype=numpy.float64 )
            gamma_sum = numpy.empty( s_num , dtype=numpy.float64 )


            for row in data:

                alpha = self.forward ( row )
                beta =  self.backward( row )

                gamma = self.single_state_prob( alpha, beta )
                ksi = self.double_state_prob( alpha, beta, row )



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


                #print("D ksi", ksi_sum )


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

            #print( numpy.exp( self._logpi ) )
            #print( numpy.exp( self._loga ) )
            #print( numpy.exp( self._logb ) )


    def meow(self):
        """Make the DtHMM to meow"""
        print('meow!')








def bw_test():
    return
    print("--bw_test--")

    print("small test:")

    A = numpy.array([[0.9,0.1],[0.4,0.6]])
    B = numpy.array([[0.9,0.1],[0.2,0.8]])
    pi = numpy.array( [0.8,0.2] )
    hmm = DtHMM(A,B,pi)

    data = numpy.array([[0,1,1]]);
    hmm.baum_welch( data )

    print( hmm.viterbi( data[0] ) )

    print("big test")

    print( get_random_vector( 5 ) )

    A = numpy.array([[0.9,0.07,0.03],[0.2,0.6,0.2],[0.15,0.05,0.8]])
    B = numpy.array([[0.9,0.005,0.07,0.005,0.02],[0.3,0.2,0.4,0.04,0.06],[0.5,0.03,0.03,0.14,0.3] ])
    pi = numpy.array( [0.8,0.15,0.05] )
    hmm = DtHMM(A,B,pi)


    num = 50
    data_len = 50
    s = numpy.empty( (num, data_len), dtype=int )
    e = numpy.empty( (num, data_len), dtype=int )

    for i in range(num):
        s[i], e[i] = hmm.generate( data_len )
        print( e[i] )

    hmmr = DtHMM( A,B,pi )
    #hmmr = DtHMM( *get_random_parameters(3,5) )
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

    bw_test()


if __name__ == "__main__":
    main()

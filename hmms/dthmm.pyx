"""
Discrete-time hidden Markov model class
"""

#
# Authors: Lukas Lopatovsky, Mai 2017
#


import numpy
import random
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
        """Initialize the DtHMM by given parameters.
        A : (n,n) ndarray
            transition probabilities matrix for (n) hidden states
        B : (n,m) ndarray
            probability matrix of (m) observation symbols being emitted by (n) hidden state
        Pi : (n) ndarray
            vector of initial probabilities
        """
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

    cpdef float_t emission_estimate(self, numpy.ndarray[int_t, ndim=1] emissions ):
        """From given emission sequence calculate the likelihood estimation given model parameters"""
        return  self.log_sum( self.forward( emissions )[-1,:] )

    cpdef float_t data_estimate( self, emissions):
        """From the set of given emission sequences in the data calculate their likelihood estimation given model parameters
           Emission sequences can be given as numpy matrix or list of numpy vectors
        """

        cdef numpy.ndarray[int_t, ndim=1] row
        cdef float_t sm = 0
        cdef int inf_cnt = 0

        for row in emissions:
            score = self.emission_estimate( row )
            if numpy.isinf(score):
                inf_cnt = inf_cnt + 1
            else:
                sm += self.emission_estimate( row )

        return sm, inf_cnt

    cpdef float_t full_data_estimate( self, state_seqs, emissions ):
        """From the set of given state and emission sequences in the data calculate their likelihood estimation given model parameters
           Emission and state sequences can be given as numpy matrix or list of numpy vectors
        """
        cdef numpy.ndarray[int_t, ndim=1] e,s
        cdef float_t sm = 0

        for  s,e in zip( state_seqs, emissions ):
            sm += self.estimate( s, e )
        return sm

    cpdef float_t estimate(self, numpy.ndarray[int_t, ndim=1] states, numpy.ndarray[int_t, ndim=1] emissions):
        """Calculate the probability of state and emission sequence given the current parameters.
           Return logaritmus of probabilities.
        """
        cdef numpy.ndarray[float_t, ndim=2] loga = self._loga
        cdef numpy.ndarray[float_t, ndim=2] logb = self._logb
        cdef numpy.ndarray[float_t, ndim=1] logpi = self._logpi
        cdef int i, s, size, states_num
        cdef float_t prob  #it is log probability

        size = emissions.shape[0]
        states_num = self._loga.shape[0]

        prob = logpi[ states[0] ] + logb[ states[0], int(emissions[0]) ]

        for i in range(1,size):
            prob += loga[states[i-1],states[i]]
            prob += logb[states[i],int(emissions[i])]

        return prob


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
        cdef numpy.ndarray[float_t, ndim=2] alpha = numpy.empty( (size,states_num), dtype=numpy.float64 )

        alpha[0,:] = logpi + logb[:, int(emissions[0]) ]
        for i in range(1,size):
            for s in range(states_num):

                alpha[i,s] = self.log_sum( alpha[i-1,:]+ loga[:,s] )

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
        cdef numpy.ndarray[float_t, ndim=2] beta = numpy.empty( (size,states_num), dtype=numpy.float64 )

        beta[-1,:] = 0  #log(1) = 0
        for i in range(size-2, -1,-1):
            for s in range(states_num):
                beta[i,s] = self.log_sum( beta[i+1,:] + loga[s,:] + logb[:, int(emissions[i+1]) ] )

        return beta

    cpdef viterbi(self, numpy.ndarray[int_t, ndim=1] e_seq):
        """
        From given emission sequence and parameters calculate the most likely state sequence
        Parameters
        ----------
        e_seq:  ndarray, int
                observation (emission) symbols sequence
        Returns
        -------
        (max_p, path) :  max_p: probability of the most likely state sequence
                         path: most likely state sequence

        """

        cdef numpy.ndarray[float_t, ndim=2] loga = self._loga
        cdef numpy.ndarray[float_t, ndim=2] logb = self._logb
        cdef numpy.ndarray[float_t, ndim=1] logpi = self._logpi
        cdef int i, s, size, states_num,
        cdef float_t max_p

        size = e_seq.shape[0]
        states_num = self._loga.shape[0]
        cdef numpy.ndarray[float_t, ndim=2] delta = numpy.empty( (size,states_num), dtype=numpy.float64 ) #numpy.zeros( (size, states_num ))
        cdef numpy.ndarray[int_t, ndim=2] psi = numpy.empty( (size,states_num), dtype=numpy.int ) #numpy.zeros( (size, states_num ))

        delta[0,:] = logpi + logb[:, int(e_seq[0]) ]
        psi[0,:] = 0
        for i in range(1,size):
            for s in range(states_num):

                delta[i,s] = delta[i-1,0] + loga[0,s]
                psi[i,s] = 0

                for r in range(1,states_num):
                    if delta[i,s] < delta[i-1,r] + loga[r,s]:
                        delta[i,s] = delta[i-1,r] + loga[r,s]
                        psi[i,s] = r

                delta[i,s] += logb[s,e_seq[i]]

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

        return ksi  #Note: actually for use in Baum welch algorithm, it wouldn't need to store whole array.

    cdef _seqs_check( self, seqs, num , error_string ):

        mx = 0
        for s in seqs:
            mx = max( mx, numpy.max(s) )
        if mx >= num:
                raise ValueError( error_string, mx+1," vs ", num )

    cpdef maximum_likelihood_estimation( self, s_seqs, e_seqs ):
        """
        Given dataset of state and emission sequences estimate the most likely parameters.
        Parameters
        ----------
        s_seqs : 2D ndarray or list of ndarrays, int
                 hidden states sequences
        e_seqs:  2D ndarray or list of ndarrays, int
                 observation (emission) symbols sequences
        """

        self._seqs_check( s_seqs,  self._logb.shape[0], "Data has more hidden states than model. " )
        self._seqs_check( e_seqs,  self._logb.shape[1], "Data has more observation symbols than model. " )


        cdef numpy.ndarray[int_t, ndim=1] sum_0, sum_last, sum_all, ss, es
        cdef numpy.ndarray[int_t, ndim=2] sum_move, sum_emit

        cdef int s_num = self._logb.shape[0]  #number of states
        cdef int o_num = self._logb.shape[1]  #number of possible observation symbols (emissions)
        cdef int seq_num,it


        if isinstance(s_seqs, list): seq_num = len(s_seqs)  #list of numpy vectors
        else: seq_num = s_seqs.shape[0]

        sum_0 =    numpy.zeros  ( s_num , dtype=numpy.int64)
        sum_last = numpy.zeros  ( s_num , dtype=numpy.int64)
        sum_all =  numpy.zeros  ( s_num , dtype=numpy.int64)
        sum_move = numpy.zeros( (s_num,s_num ) , dtype=numpy.int64)
        sum_emit = numpy.zeros( (s_num,o_num ) , dtype=numpy.int64)

        for ss,es in zip( s_seqs, e_seqs):

            sum_0[ss[0]]+= 1
            sum_all[ss[0]]+= 1
            sum_emit[ ss[0], es[0] ]+=1
            sum_last[ ss[-1] ]+=1

            for it in range(1, ss.size ):

                sum_all[ ss[it] ]+=1
                sum_move[ ss[it-1], ss[it] ]+=1
                sum_emit[ ss[it], es[it] ]+=1

        self._logpi = numpy.log( sum_0 / seq_num )
        self._loga  = numpy.log( (sum_move.T / (sum_all-sum_last ) ).T )
        self._logb  = numpy.log( (sum_emit.T / sum_all).T )

    cpdef states_confidence( self, e_seq ):
        """Given emission sequence, return probabilities that emission is generated by a state, for every time and every state."""
        return self.single_state_prob( self.forward ( e_seq ), self.backward( e_seq ) )

    def baum_welch( self, e_seqs, iterations = 10, **kvargs ):
        """
        Estimate parameters by Baum-Welch algorithm

        Parameters
        ----------
        e_seqs:  2D ndarray or list of ndarrays
              observation (emission) symbols sequences
        iterations: Optional[int]
                    number of algorithm iterations
        **est :  boolean
                 if True return the vector of estimations for every iteration
                 default: False
        Returns
        -------
        graph : (iterations + 1) ndarray
                if **est== True
                None otherwise
        References
        ----------
        .. [1] Rabiner, L. R.: A tutorial on hidden Markov models and selected applic-
               ations in speech recognition. Proceedings of the IEEE, volume 77, no. 2,
               1989: pp. 257–286.
        """
        if 'est' in kvargs:
            if kvargs['est'] == True:
                return self._baum_welch( e_seqs, True, iterations )

        self._baum_welch( e_seqs, False, iterations )

    cpdef _baum_welch(self, data, int est, iterations = 10 ):
        """
        Estimate parameters by Baum-Welch algorithm.
        Called internally by baum_welch function.
        """

        self._seqs_check( data,  self._logb.shape[1], "Data has more observation symbols than model. " )

        cdef numpy.ndarray[float_t, ndim=1] gamma_sum, pi_sum, gamma_full_sum, gamma_part_sum
        cdef numpy.ndarray[int_t, ndim=1] row
        cdef numpy.ndarray[float_t, ndim=2] alpha, beta, gamma, ksi_sum, obs_sum
        cdef numpy.ndarray[float_t, ndim=3] ksi

        cdef int s_num = self._logb.shape[0]  #number of states
        cdef int o_num = self._logb.shape[1]  #number of possible observation symbols (emissions)
        cdef int i,j,t,it, seq_num

        if isinstance(data, list): seq_num = len(data)  #list of numpy vectors
        else: seq_num = data.shape[0]                   #numpy matrix


        if est:
            graph = numpy.zeros(iterations+1)

        for it in range( iterations ):



            print("iteration ", it+1, "/", iterations )

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

                if est:
                    graph[it] += self.log_sum( alpha[-1,:] )


                #expected number of being in state i in time 0
                for i in range( s_num ):
                    pi_sum[i] = self.log_sum_elem( pi_sum[i], gamma[0,i] )


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
            self._logpi = pi_sum - numpy.log( seq_num )  #average
            #transition matrix estimation
            self._loga = (ksi_sum.T - gamma_part_sum).T
            #observetion symbol emission probabilities estimation
            self._logb = (obs_sum.T - gamma_full_sum).T

            self._loga[s_num - 1:] = 0
            self._loga[s_num - 1, s_num - 1] = 1


        if est:
            score, inf_count = self.data_estimate( data)
            graph[iterations] = self.data_estimate( data)
            graph[iterations] = score + -100 * inf_count
            return graph


    def meow(self):
        """Make the DtHMM to meow"""
        print('meow!')


def main():
    my_hmm = DtHMM()
    my_hmm.meow()

if __name__ == "__main__":
    main()

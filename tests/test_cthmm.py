import hmms
import pytest
import numpy
import scipy.linalg

from common import *

@pytest.fixture
def cthmm():
    """Parameters for cthmm of 3 hidden states and 3  output variables"""
    Q = numpy.array( [[-0.375,0.125,0.25],[0.25,-0.5,0.25],[0.25,0.125,-0.375]] )
    B = numpy.array( [[0.8,0.05,0.15],[0.05,0.9,0.05],[0.2,0.05,0.75]] )
    Pi = numpy.array( [0.6,0,0.4] )
    return hmms.CtHMM(Q,B,Pi)

@pytest.fixture
def dthmm( cthmm ):
    """The discrete model, created so it behaves identical to the given continuous."""
    return hmms.DtHMM( *cthmm.get_dthmm_params() )

@pytest.fixture
def cthmm_zeros():
    """Parameters for cthmm of 3 hidden states and 3  output variables, with zeros is Q"""
    Q = numpy.array( [[-0.125,0.125,0.0],[0.25,-0.5,0.25],[0.0,0.125,-0.125]] )
    B = numpy.array( [[0.8,0.05,0.15],[0.05,0.9,0.05],[0.2,0.05,0.75]] )
    Pi = numpy.array( [0.6,0,0.4] )
    return hmms.CtHMM(Q,B,Pi)

def test_zeros( cthmm_zeros ):

    t = numpy.array([ [0,5,8,9,14,19],[0,3,6,7,12,13],[0,5,6,11,14,19] ])
    e = numpy.array([ [0,0,1,0,1,0],[0,1,2,0,1,0],[2,2,0,1,0,2] ])

    cthmm_zeros.baum_welch( t, e, 10 ,method="soft" )

    result = hmms.CtHMM( numpy.array( [[-0.21327285,  0.21327285,  0.        ],
                                       [ 0.42250406, -0.6310883 ,  0.20858425],
                                       [ 0.        ,  0.33834679, -0.33834679]]),
                         numpy.array([[ 0.5696601 ,  0.20239348,  0.22794642],
                                      [ 0.28613018,  0.71024895,  0.00362086],
                                      [ 0.56025733,  0.03147241,  0.40827027]]),
                         numpy.array([ 0.41600342,  0.        ,  0.58399658])
    )

    assert compare_parameters_no_sort( result, cthmm_zeros)


@pytest.mark.parametrize(
    ['data_num', 'data_len'],
    [(1, 100),
     (3, 100),
     (20, 20),
     (40, 2), ],
)
def test_compare_state_probs_with_discrete( data_num, data_len,dthmm ):
    """Test will run algorithms for counting state probability, determinically with the same initialization for both models"""
    t,_,e = dthmm.generate_data( (data_num,data_len), times=True )

    ct = hmms.CtHMM.random(3,3)
    dt = hmms.DtHMM( *ct.get_dthmm_params() )

    assert compare_parameters_no_sort( dt,  hmms.DtHMM( *ct.get_dthmm_params() ) )

    row = e[0]
    trow = t[0]

    #ct
    alpha = ct.forward ( trow, row )
    beta =  ct.backward( trow, row )
    gamma = ct.single_state_prob( alpha, beta )
    ksi = ct.double_state_prob( alpha, beta, trow, row )
    #dt
    d_alpha = dt.forward ( row )
    d_beta =  dt.backward( row )
    d_gamma = dt.single_state_prob( d_alpha, d_beta )
    d_ksi = dt.double_state_prob( d_alpha, d_beta, row )

    assert float_equal_mat( gamma, d_gamma  )
    assert float_equal_mat( ksi[0], d_ksi  )

@pytest.mark.parametrize(
    ['data_num', 'data_len'],
    [(1, 100),
     (3, 100),
     (20, 20),
     (40, 2), ],
)
def test_estimate( data_num, data_len,dthmm ):
    """Test will run algorithms for counting state probability, determinically with the same initialization for both models"""
    t,s,e = dthmm.generate_data( (data_num,data_len), times=True )

    ct = hmms.CtHMM.random(3,3)
    dt = hmms.DtHMM( *ct.get_dthmm_params() )

    cte = ct.estimate(s[0],t[0],e[0])
    dte = dt.estimate(s[0],e[0])

    assert ( cte == dte )


@pytest.mark.parametrize("t,e,num", [
    (numpy.array([ [0,5,8,9,14,19],[0,3,6,7,12,13],[0,5,6,11,14,19] ]),
     numpy.array([ [0,0,1,0,1,0],[0,1,2,0,1,0],[2,2,0,1,0,2] ]),
     3
    ),
    (numpy.array([ [0,1,8,16,19,29],[0,2,60,77,120,133],[0,50,61,70,77,79] ]),
     numpy.array([ [0,0,1,0,1,0],[0,1,2,0,1,0],[2,2,0,1,0,2] ]),
     13
    ),
    (numpy.array([ [0,1,8,16,19,29],[0,2,60,77,120,133],[0,50,61,70,77,79] ]),
     numpy.array([ [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0] ]),
     13
    ),
    (numpy.array([ [0,1,8,16,19,29],[0,2,60,77,120,133],[0,50,61,70,77,79] ]),
     numpy.array([ [1,1,1,1,1,0],[1,1,0,0,0,0],[1,1,0,0,0,0] ]),   #impossibility of init model
     13
    )

])
def test_time_intervals_mapping( t,e,num, cthmm ):
    """test if the time intervals compress to uniq intervals correctly.
    Training data of various time intervals.
    Data are created in the way so it encompass zero probability -> -inf logprobability.
    """
    cthmm.baum_welch( t, e , 3)

    assert cthmm.time_n == num

@pytest.fixture
def train_data():
    data = ( numpy.array([ [ 0,  1,  2,  3,  6,  8, 13, 14, 15, 16],
                           [ 0,  1,  2,  3,  5,  8, 10, 14, 17, 20],
                           [ 0,  6,  9, 14, 17, 21, 22, 25, 29, 30],
                           [ 0,  1,  2,  5,  8, 10, 11, 14, 16, 17],
                           [ 0,  1,  3,  5,  7,  9, 12, 15, 16, 22 ] ] ),
             numpy.array([ [0, 0, 0, 0, 2, 1, 1, 1, 2, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 2],
                           [0, 0, 1, 0, 2, 0, 0, 2, 2, 2],
                           [0, 0, 0, 2, 1, 2, 2, 2, 2, 0],
                           [2, 0, 0, 2, 0, 2, 2, 0, 0, 0] ] )
           )
    return data

@pytest.fixture
def out_params():
    """Parameters obtained from cthmm after train at """
    Q = numpy.array( [[-0.248751,0.090448,0.158303],
	                  [0.173265,-0.738227,0.564962],
                      [0.210508,0.165507,-0.376015]] )
    B = numpy.array( [[0.883138,0.015511,0.101351],
                      [0.005337,0.872273,0.122390],
                      [0.207833,0.015863,0.776304]] )
    Pi = numpy.array( [0.999943,0.000000,0.000057] )
    return hmms.CtHMM(Q,B,Pi)

@pytest.mark.parametrize(
    ['h_states', 'o_symbols', 'data_num', 'data_len'],
    [(3, 3, 20, 20),
     (2, 1, 20, 20),
     (2, 10, 20, 20),
     (10, 1, 20, 20),
     (5, 5, 1, 200),
     (5, 7, 3, 50),
     (5, 3, 3, 50),
     (5, 5, 50, 2),
     (20, 20, 5, 10),
    ],
)
def test_growing_likelihood(h_states, o_symbols, data_num, data_len):
    """The likelihood in the EM algorithm had always to grow"""

    cthmm = hmms.CtHMM.random( h_states, o_symbols )
    t, e = cthmm.generate_data( ( data_num, data_len ) )

    cthmm_t = hmms.CtHMM.random( h_states, o_symbols )

    dl = cthmm_t.baum_welch( t,e,15, est=True )

    #print("~~~~~~<")
    #print(dl)

    assert float_equal_mat( dl, numpy.sort( dl )  )


def test_baum_welch( train_data, cthmm, out_params ):
    """This is just the consistency test, do not ensure right computations"""
    t,e = train_data

    cthmm.baum_welch( t,e,20 )

    assert compare_parameters_no_sort( out_params,  cthmm, 1e-5 )




#TODO pridaj ako test na prepare matrices
#    def zmaz_ma( self, times ):
#        self._prepare_matrices_pt( numpy.array( [times] ) )
#        for i in range ( 1, times.shape[0] ):
#            interval = times[i] - times[i-1]
#            print("i:",interval)
#            print( numpy.asarray( self._pt[ self.tmap[ interval ] ]  ) )
#
#        self._prepare_matrices_n_exp()

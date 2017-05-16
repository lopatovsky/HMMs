import hmms
import pytest
import numpy

from common import *

@pytest.fixture
def small_hmm():
    """Create small DtHMM and basic emission sequence for testing of basic functionality"""
    A = numpy.array([[0.9,0.1],[0.4,0.6]])
    B = numpy.array([[0.9,0.1],[0.2,0.8]])
    pi = numpy.array( [0.8,0.2] )
    hmm = hmms.DtHMM(A,B,pi)

    emissions = numpy.array([0,1])
    return ( hmm, emissions )

@pytest.fixture
def small_hmm2():
    """Create small DtHMM  for testing of basic functionality"""
    A = numpy.array([[0.9,0.1],[0.4,0.6]])
    B = numpy.array([[0.75,0.25],[0,1.0]])
    pi = numpy.array( [0.8,0.2] )
    hmm = hmms.DtHMM(A,B,pi)

    return hmm

@pytest.fixture
def short_emission( small_hmm ):
    """Return DtHMM and medium emission sequence"""
    hmm, em = small_hmm
    em = numpy.array([0,1,1])
    return (hmm, em)


@pytest.fixture
def medium_emission( small_hmm ):
    """Return DtHMM and medium emission sequence"""
    hmm, em = small_hmm
    em = numpy.array([0,1,0,1,1])
    return (hmm, em)

@pytest.fixture
def long_emission( small_hmm ):
    """Return DtHMM and longer emission sequence"""
    hmm, em = small_hmm
    em = numpy.array([0,0,0,1,1,0,1,1,0,0,0,0,0,1,0,0,1,1,0,0,0,1,0,1,1,1,1,0,0,1,1,1])
    return (hmm, em)

def test_forward( small_hmm ):
    """Test forward algorithm"""
    hmm, em = small_hmm

    A = numpy.array( [[0.72,0.04],[0.0664,0.0768]] )
    X = numpy.exp( hmm.forward(em) )

    assert float_equal_mat( A, X )

def test_backward( small_hmm ):
    """Test backward algorithm"""
    hmm, em = small_hmm

    A = numpy.array( [[0.17,0.52],[1,1]] )
    X = numpy.exp( hmm.backward(em) )

    assert float_equal_mat( A, X )

def test_emission_estimate( small_hmm ):
    """Test emission_estimate function"""
    hmm, em = small_hmm

    a = 0.1432
    x = numpy.exp( hmm.emission_estimate(em) )

    assert float_equal( a, x )

def test_estimate( small_hmm ):
    """Test emission_estimate function"""
    hmm, em = small_hmm
    states = numpy.array([0,0])

    a = 0.0648
    x = numpy.exp( hmm.estimate(states,em) )

    assert float_equal( a, x )

def test_random_vector_and_log_sum( small_hmm ):
    """test if random vector sum to one by using log_sum function"""
    hmm, em = small_hmm

    size = 1234
    vec = hmms.DtHMM.get_random_vector(size)
    a = numpy.exp( hmm.log_sum( numpy.log(vec)  ) )

    assert float_equal( a, 1 )

def test_random_vector_and_log_sum_elem( small_hmm ):
    """test if random vector sum to one by using repeatedly log_sum_elem function"""
    hmm, em = small_hmm

    size = 1234
    vec = hmms.DtHMM.get_random_vector ( size )
    a = 0
    for num in vec:
        a = numpy.exp(  hmm.log_sum_elem( numpy.log(a), numpy.log(num) ) )

    assert float_equal( a, 1 )

def test_viterbi_p( medium_emission ):
    """Test viterbi algorithm probability of most likely sequence"""
    hmm, em = medium_emission

    p, seq = hmm.viterbi( em )
    out_p = 0.0020155392

    assert float_equal( numpy.exp(p), out_p )

def test_viterbi_seq( medium_emission ):
    """Test viterbi algorithm sequence"""
    hmm, em = medium_emission

    p, seq = hmm.viterbi( em )
    out_seq = numpy.array([0,0,0,1,1])

    assert float_equal_mat( seq, out_seq )

def test_viterbi_long_seq( long_emission ):
    """Test viterbi algorithm sequence"""
    hmm, em = long_emission

    p, seq = hmm.viterbi( em )
    out_seq = numpy.array( [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1] )

    assert float_equal_mat( seq, out_seq )

@pytest.fixture
def ab( short_emission ):
    """Return outputs from forward and backward algorithms"""
    hmm, em = short_emission
    alpha = hmm.forward(em)
    beta = hmm.backward(em)
    return (alpha, beta)

def test_single_state_prob( short_emission, ab ):
    """Test single_state_prob function"""
    hmm, em = short_emission
    alpha, beta = ab
    gamma = hmm.single_state_prob( alpha, beta )
    gamma_out = numpy.array( [[ 0.79978135, 0.20021865],[ 0.22036545,0.77963455],[ 0.17663595,0.82336405]] )

    assert float_equal_mat( numpy.exp(gamma), gamma_out )

def test_double_state_prob( short_emission, ab ):
    """Test double_state_prob function"""
    hmm, em = short_emission
    alpha, beta = ab
    ksi = hmm.double_state_prob( alpha, beta, em )[1,:]
    ksi_out = numpy.array( [[ 0.11666406,0.10370139],[ 0.05997189,0.71966266]] )

    assert float_equal_mat( numpy.exp(ksi), ksi_out )


@pytest.fixture
def small_random_hmm():
    """Create random hmm of two hidden states and two output varaibles."""
    return hmms.DtHMM.random( 2,2 )

@pytest.fixture
def hmm_small_out():
    """Desired training output for sequence [[0,1,1]]"""
    A = numpy.array([[0,1],[0,1]])
    B = numpy.array([[1,0],[0,1]])
    pi = numpy.array( [1,0] )
    return hmms.DtHMM(A,B,pi)

@pytest.fixture
def hmm_cycle_out():
    """Desired training output for sequence [[0,1,1]]"""
    A = numpy.array([[0,1],[1,0]])
    B = numpy.array([[1,0],[0,1]])
    pi = numpy.array( [0.5,0.5] )
    return hmms.DtHMM(A,B,pi)

def test_baum_welch_small( small_random_hmm, hmm_small_out ):
    """Test if baum_welch algorithm converge to the right parameters"""
    hmm = small_random_hmm
    data = numpy.array([[0,1,1]]);
    hmm.baum_welch( data , 200 )

    assert compare_parameters( hmm, hmm_small_out, 1e-2 )

def test_baum_welch_small_multiple_data( small_random_hmm, hmm_small_out ):
    """Test if baum_welch algorithm converge to the right parameters"""
    hmm = small_random_hmm
    data = numpy.array([[0,1,1],[0,1,1],[0,1,1],[0,1,1],[0,1,1]]);
    hmm.baum_welch( data , 20 )

    assert compare_parameters( hmm, hmm_small_out, 1e-2 )


def test_mle( small_hmm2 ):
    """Test if maximum_likelihood_estimation function return correct values"""
    hmm = hmms.DtHMM.random( 2,2 )
    ss = numpy.array( [
       [0,0,0,0,0,0],
       [0,0,0,0,0,1],
       [0,1,1,1,0,0],
       [1,1,0,0,0,0],
       [0,0,0,0,0,0]
    ] )

    es = numpy.array( [
       [0,1,1,0,0,0],
       [0,0,0,1,0,1],
       [0,1,1,1,1,0],
       [1,1,0,0,0,0],
       [0,0,1,0,1,0]
    ] )

    hmm.maximum_likelihood_estimation( ss, es )

    print( hmm.a )
    print( hmm.b )
    print( hmm.pi )


    assert compare_parameters( hmm, small_hmm2, 1e-7 )

def test_mle_big( ):

    hmm = hmms.DtHMM.random(2,5)
    states, seq = hmm.generate( 1000 );
    hmm.maximum_likelihood_estimation( [states], [seq] )



    hmm = hmms.DtHMM.random(2,5)
    states, seq = hmm.generate_data( ( 1,1000 ) );
    hmm.maximum_likelihood_estimation(states, seq)

@pytest.mark.parametrize(
    ['h_states', 'o_symbols', 'data_num', 'data_len'],
    [(3, 3, 20, 20),
     (2, 1, 20, 20),
     (2, 10, 20, 20),
     (10, 1, 20, 20),
     (5, 5, 1, 50),
     (5, 7, 3, 25),
     (5, 3, 3, 25),
     (5, 5, 25, 2),
     (10, 10, 5, 5),
    ],
)
def test_growing_likelihood_mle(h_states, o_symbols, data_num, data_len):
    """The likelihood in the MLE algorithm had always to be bigger or equal to original sequence"""
    dhmm = hmms.DtHMM.random(h_states,o_symbols)
    s_seqs , e_seqs = dhmm.generate_data( (data_num,data_len) )

    dhmm_r = hmms.DtHMM.random(h_states,o_symbols)

    dhmm_r.maximum_likelihood_estimation(s_seqs,e_seqs)

    log_est =     dhmm.full_data_estimate  ( s_seqs, e_seqs )
    log_est_mle = dhmm_r.full_data_estimate( s_seqs, e_seqs )

    assert log_est_mle >= log_est


import cthmm
import pytest
import numpy

EPS = 1e-7

@pytest.fixture
def small_hmm():
    """Create small HMM and emission sequence for testing of basic functionality"""
    A = numpy.array([[0.9,0.1],[0.4,0.6]])
    B = numpy.array([[0.9,0.1],[0.2,0.8]])
    pi = numpy.array( [0.8,0.2] )
    hmm = cthmm.HMM(A,B,pi)

    emissions = numpy.array([0,1])
    return ( hmm, emissions )


def test_froward( small_hmm ):
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

def test_random_vector_and_log_sum( small_hmm ):
    """test if random vector sum to one by using log_sum function"""
    hmm, em = small_hmm

    size = 1234
    vec = cthmm.get_random_vector(size)
    a = numpy.exp( hmm.log_sum( numpy.log(vec)  ) )

    assert float_equal( a, 1 )

def test_random_vector_and_log_sum_elem( small_hmm ):
    """test if random vector sum to one by using repeatedlly log_sum_elem function"""
    hmm, em = small_hmm

    size = 1234
    vec = cthmm.get_random_vector ( size )
    a = 0
    for num in vec:
        a = numpy.exp(  hmm.log_sum_elem( numpy.log(a), numpy.log(num) ) )

    assert float_equal( a, 1 )


def float_equal( a , b ):
    """Compare two floats with possible error EPS"""
    return numpy.fabs(a-b) < EPS

def float_equal_mat( A , B ):
    """Takes two numpy arrays or vectors and tells if they are equal (with possible EPS error caused by double imprecision)"""
    for a,b in zip( A.flatten(), B.flatten() ):
        if numpy.fabs(a-b) > EPS : return False
    return True


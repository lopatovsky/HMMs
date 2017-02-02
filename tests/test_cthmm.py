import hmms
import pytest
import numpy

EPS = 1e-7

@pytest.fixture
def small_hmm():
    """Create small DtHMM and basic emission sequence for testing of basic functionality"""
    A = numpy.array([[0.9,0.1],[0.4,0.6]])
    B = numpy.array([[0.9,0.1],[0.2,0.8]])
    pi = numpy.array( [0.8,0.2] )
    hmm = hmms.DtHMM(A,B,pi)

    emissions = numpy.array([0,1])
    return ( hmm, emissions )

def test_froward( small_hmm ):
    """Test forward algorithm"""
    hmm, em = small_hmm

    A = numpy.array( [[0.72,0.04],[0.0664,0.0768]] )
    X = numpy.exp( hmm.forward(em) )

    assert float_equal_mat( A, X )


###Common methods TODO - move in separate file


def compare_parameters( m1, m2, eps ):
    """Compare whole hmm parameters"""

    #Notice: sort_rows is needed, because any permutation of hidden states is acceptable
    ok  = float_equal_mat( sort_mat(m1.a),  sort_mat(m2.a), eps )
    ok &= float_equal_mat( sort_mat(m1.b),  sort_mat(m2.b), eps )
    ok &= float_equal_mat( numpy.sort(m1.pi), numpy.sort(m2.pi),eps )

    return ok

def get_hash( a ):
    """Return hash of the numpy vector or the number itself, if it is float"""
    val = 0
    for i in a:
        val += i
        val *= 10
    return val

def sort_rows( m ):
    """Sort rows in numpy array in some deterministic order"""
    sm = numpy.zeros( m.shape[0] )
    for i,a in enumerate(m):
        sm[i] = get_hash( a )
    return m[ numpy.argsort( sm ),:]

def sort_mat( m ):
    """Sort matrix in the way, so the all hidden state permutation will form the same matrix"""
    print("m")
    print(m)
    m = sort_rows(m)
    print(m)
    m = sort_rows(m.T)
    print(m)
    return m

def float_equal( a , b , eps = EPS):
    """Compare two floats with possible error EPS"""
    print(a,b)
    return numpy.fabs(a-b) < eps

def float_equal_mat( A , B, eps = EPS):
    """Takes two numpy arrays or vectors and tells if they are equal (with possible EPS error caused by double imprecision)"""
    print("#####Compare_matrices#####")
    print(A)
    print("-"*10)
    print(B)
    print("#"*10)
    for a,b in zip( A.flatten(), B.flatten() ):
        if numpy.fabs(a-b) > eps :
            print("Do not equal!")
            return False
    return True


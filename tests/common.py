import hmms
import pytest
import numpy

EPS = 1e-7

def compare_parameters_no_sort( m1, m2, eps = EPS ):
    """Compare whole hmm parameters"""

    #Notice: sort_rows is needed, because any permutation of hidden states is acceptable
    if hasattr(m1, 'a'):
        ok  = float_equal_mat( m1.a,  m2.a, eps )
    else:
        ok  = float_equal_mat( m1.q,  m2.q, eps )

    ok &= float_equal_mat( m1.b,  m2.b, eps )
    ok &= float_equal_mat( m1.pi, m2.pi,eps )

    return ok


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

def numpy_to_list( A ):
    l = []
    for row in A:
        l.append(row)

    return l

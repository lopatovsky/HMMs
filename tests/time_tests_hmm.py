import hmms
import numpy
import time
import random

def get_random_data(  n, m, num, data_len ):

    hmm = hmms.DtHMM.random(n,m)

    e = numpy.empty( (num, data_len), dtype=int )

    for i in range(num):
        _, e[i] = hmm.generate( data_len )
        print( e[i] )

    print( hmm.pi )

    return e


def make_time_test( n, m, num, data_len, it_num ):
    """Time test, parameters: n-> hidden states, m-> output symbols, num-> number of data vectors, data_len -> length of data vectors,
                              it_num -> number of iterations for Baum-welch algorithm
    """

    data = get_random_data( n, m, num, data_len )
    hmm = hmms.DtHMM.random( n,m )

    print( hmm.pi )

    start_time = time.time()

    hmm.baum_welch( data, it_num )

    print("Time test size(",n,m,") - ", num, "vectors of len = ", data_len, ",traied in", it_num, "iterations." )
    print("--- %s seconds ---" % (time.time() - start_time))


def main():

    random.seed(42)

    make_time_test( 10,10,50,50 ,10 )



if __name__ == "__main__":
    main()

import hmms
import numpy as np
import time
import random
import matplotlib.pyplot as plt

def get_random_data_dt(  n, m, num, data_len ):

    hmm = hmms.DtHMM.random(n,m)

    e = numpy.empty( (num, data_len), dtype=int )

    for i in range(num):
        _, e[i] = hmm.generate( data_len )
        print( e[i] )

    return e


def make_time_test_dt( n, m, num, data_len, it_num ):
    """Time test, parameters: n-> hidden states, m-> output symbols, num-> number of data vectors, data_len -> length of data vectors,
                              it_num -> number of iterations for Baum-welch algorithm
    """

    data = get_random_data_dt( n, m, num, data_len )
    hmm = hmms.DtHMM.random( n,m )

    start_time = time.time()

    hmm.baum_welch( data, it_num )

    print("Time test size(",n,m,") - ", num, "vectors of len = ", data_len, ",traied in", it_num, "iterations." )
    print("--- %s seconds ---" % (time.time() - start_time))


def get_random_data_ct(  n, m, num, data_len ):

    hmm = hmms.CtHMM.random(10,10)
    return hmm.generate_data( (num,data_len) )


def make_time_test_ct( n, m, num, data_len, it_num ):
    """Time test, parameters: n-> hidden states, m-> output symbols, num-> number of data vectors, data_len -> length of data vectors,
                              it_num -> number of iterations for Baum-welch algorithm
    """

    times,data = get_random_data_ct( n, m, num, data_len )
    hmm = hmms.CtHMM.random( n,m )

    start_time = time.time()

    hmm.baum_welch( times, data, it_num )

    print("Time test size(",n,m,") - ", num, "vectors of len = ", data_len, ",traied in", it_num, "iterations." )
    print("--- %s seconds ---" % (time.time() - start_time))

    print( "Estimation", numpy.exp( hmm.data_estimate(times,data) ) )
    hmm.print_ts()


def cd_convergence_ex():


    #We will start by defining the continuous time model. For that, who have read the previous section, it will be the familiar.

    Q = np.array( [[-0.375,0.125,0.25],[0.25,-0.5,0.25],[0.25,0.125,-0.375]] )
    B = np.array( [[0.8,0.05,0.15],[0.05,0.9,0.05],[0.2,0.05,0.75]] )
    Pi = np.array( [0.6,0,0.4] )

    chmm = hmms.CtHMM( Q,B,Pi )
    hmms.print_parameters( chmm )

    #We can simply create discrete model with equivalent parameters, using function get_dthmm_params.
    #By default, it will create the model with transition probabilities equal to one time unit probability transition in continuous model. You can pass the optional parameter for different time steps.

    dhmm = hmms.DtHMM( *chmm.get_dthmm_params() )
    hmms.print_parameters( dhmm )

    #We can let the disrete model to generate the data sufficient for both models by passing the times parameter as True.

    t,_,e = dhmm.generate_data( (50,50), times=True )   # The free space in the return triple is for the state sequences, we do not need them for the training

    #We can compare the estimation of the data, using both of the model. (They should be the same.)

    creal = chmm.data_estimate(t,e)
    dreal = dhmm.data_estimate(e)
    print("Data estimation by continuous model:", creal)
    print("Data estimation by discrete model:  ", dreal)

    #Now we will create two equivalent random models.

    ct = hmms.CtHMM.random(3,3)
    dt = hmms.DtHMM( *ct.get_dthmm_params() )

    hmms.print_parameters( ct )
    hmms.print_parameters( dt )

    #We will train them at our dataset. (It can take a while.)

    #iter_num = 50
    #outd = dt.baum_welch( e,   iter_num, est=True )
    #outc = ct.baum_welch( t,e, iter_num, est=True )

    hidden_states = 3
    runs = 2
    iterations = 5
    out_ct = hmms.multi_train_ct( hidden_states , t, e, runs, iterations, ret='all', method='unif')
    out_dt = hmms.multi_train_dt( hidden_states , e, runs, iterations, ret='all' )



    #We can plot and compare both convergence rates. From the essence of models, the continuous model will probably converge a bit slower, but finally will reach the similar value.

    plt.plot( out_ct[1][1][:] / dreal )
    plt.plot( out_dt[1][1][:] / dreal )
    plt.plot( out_dt[0][1][:] / dreal )
    plt.plot( out_ct[0][1][:] / dreal )

    #plt.plot( outd[1:] / dreal )
    #plt.plot( outc[1:] / dreal )
    #plt.savefig('my_plot.svg')  #Optional save the figure

    plt.show()





def main():

    random.seed(42)

    #make_time_test_dt( 10,10,50,50 ,10 )
    #make_time_test_dt( 4,4,50,150  ,10 )


    ## time experiment watching the complexity of growing states number
    #for i in range(2,21,2):
    #    make_time_test_ct( i, 10, 10, 10, 10)

    cd_convergence_ex()



if __name__ == "__main__":
    main()

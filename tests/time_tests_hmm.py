import hmms
import numpy as np
import time
import numpy
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle


def distance( A, B ):

    A = A.flatten()
    B = B.flatten()

    out = 0.0

    for a,b in zip(A,B):
        out += numpy.fabs(a-b)*numpy.fabs(a-b)

    return numpy.sqrt(out)

def rel_distance( A, B ):

    A = A.flatten()
    B = B.flatten()

    mx =  max( numpy.max(A) ,  numpy.max(B) )

    out = 0.0

    for a,b in zip(A,B):
        out += numpy.fabs(a-b)*numpy.fabs(a-b)

    return numpy.sqrt(out) / mx




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


def make_time_test_ct( n, m, num, data_len, it_num, fast_val ):
    """Time test, parameters: n-> hidden states, m-> output symbols, num-> number of data vectors, data_len -> length of data vectors,
                              it_num -> number of iterations for Baum-welch algorithm
    """

    times,data = get_random_data_ct( n, m, num, data_len )
    hmm = hmms.CtHMM.random( n,m )

    start_time = time.time()

    hmm.baum_welch( times, data, it_num , fast= fast_val )

    print("Time test size(",n,m,") - ", num, "vectors of len = ", data_len, ",traied in", it_num, "iterations." )
    print("--- %s seconds ---" % (time.time() - start_time))

    print( "Estimation", numpy.exp( hmm.data_estimate(times,data) ) )
    hmm.print_ts()


def get_random_data_ct_od(  n, m, num, data_len, mx_time ):
    hmm = hmms.CtHMM.random(n,m)

    _ , data = hmm.generate_data( (num,data_len) )

    times = numpy.empty( data.shape, dtype=numpy.float64 )

    for i in range( num ):
        time = 0.0
        for j in range( data_len ):
           times[i][j] = time
           if i == 0 and j == 0: time += mx_time
           else: time += random.randint(1,mx_time)

    return times, data



def make_time_test_ct_od( n, m, num, data_len, it_num, fast_val, mx_time ):
    """Time test, parameters: n-> hidden states, m-> output symbols, num-> number of data vectors, data_len -> length of data vectors,
                              it_num -> number of iterations for Baum-welch algorithm
    """

    times,data = get_random_data_ct_od( n, m, num, data_len, mx_time )
    hmm = hmms.CtHMM.random( n,m )

    start_time = time.time()

    hmm.baum_welch( times, data, it_num , fast= fast_val )

    print("Time test size(",n,m,") - ", num, "vectors of len = ", data_len, ",traied in", it_num, "iterations." )
    print("--- %s seconds ---" % (time.time() - start_time))

    print( "Estimation", numpy.exp( hmm.data_estimate(times,data) ) )
    hmm.print_ts()




def get_random_data_ct_od2(  n, m, num, data_len, mx_time ):
    hmm = hmms.CtHMM.random(n,m)

    _ , data = hmm.generate_data( (num,data_len) )

    times = numpy.empty( data.shape, dtype=numpy.float64 )

    diff = 0

    for i in range( num ):
        time = 0.0
        for j in range( data_len ):
           times[i][j] = time
           time += mx_time + diff
           diff+=1

    return times, data



def make_time_test_ct_od2( n, m, num, data_len, it_num, fast_val, mx_time ):
    """Time test, parameters: n-> hidden states, m-> output symbols, num-> number of data vectors, data_len -> length of data vectors,
                              it_num -> number of iterations for Baum-welch algorithm
    """

    times,data = get_random_data_ct_od2( n, m, num, data_len, mx_time )
    hmm = hmms.CtHMM.random( n,m )

    #print(times)

    start_time = time.time()

    hmm.baum_welch( times, data, it_num , fast= fast_val )

    #print("Time test size(",n,m,") - ", num, "vectors of len = ", data_len, ",traied in", it_num, "iterations." )
    #print("--- %s seconds ---" % (time.time() - start_time))

    #print( "Estimation", numpy.exp( hmm.data_estimate(times,data) ) )
    #hmm.print_ts()

    return (time.time() - start_time)



def cd_convergence_ex1():


    out_c = []
    out_d = []

    models = 1
    offset = 1

    for m in range(models):

        Q = np.array( [[-0.375,0.125,0.25],[0.25,-0.5,0.25],[0.25,0.125,-0.375]] )
        B = np.array( [[0.8,0.05,0.15],[0.05,0.9,0.05],[0.2,0.05,0.75]] )
        Pi = np.array( [0.6,0,0.4] )

        chmm = hmms.CtHMM( Q,B,Pi )
        #chmm = hmms.CtHMM.random( 3,3 )

        #hmms.print_parameters( chmm )

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


        #We will train them at our dataset. (It can take a while.)

        #iter_num = 50
        #outd = dt.baum_welch( e,   iter_num, est=True )
        #outc = ct.baum_welch( t,e, iter_num, est=True )

        hidden_states = 3
        runs = 10 #20
        iterations = 150
        out_dt, out_ct = hmms.multi_train_ctdt( hidden_states , t, e, runs, iterations, ret='all', method='unif')

        for ( m, a ) in out_ct:
            out_c.append(a/dreal)


        for ( m, a ) in out_dt:
            out_d.append(a/dreal)

    print("out_c")
    print(out_c)
    print("out_d")
    print(out_d)
    #We can plot and compare both convergence rates. From the essence of models, the continuous model will probably converge a bit slower, but finally will reach the similar value.



    ##LEGEND

    fig = plt.figure()
    #fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    #fig.subplots_adjust(top=0.85)
    ax.set_title('Models Comparison')

    ax.set_xlabel('iterations')
    ax.set_ylabel('performance')

    fire = mpatches.Patch(color='red', label='CT - single run')
    olive = mpatches.Patch(color='chartreuse', label='DT - single run')

    plt.legend(handles=[olive, fire ])

    ## DATA PLOT

    for i in range(runs*models):
        plt.plot( out_d[i][offset:]  , color="chartreuse" )
        plt.plot( out_c[i][offset:]  , color="red" )

    #plt.plot( outd[1:] / dreal )
    #plt.plot( outc[1:] / dreal )
    #plt.savefig('my_plot.svg')  #Optional save the figure

    ##LEGEND

    fig2 = plt.figure()
    #fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

    ax2 = fig2.add_subplot(111)
    #fig.subplots_adjust(top=0.85)
    ax2.set_title('Models Comparison')

    ax2.set_xlabel('iterations')
    ax2.set_ylabel('performance')

    red_patch = mpatches.Patch(color='red', label='CT - average')
    char_patch = mpatches.Patch(color='chartreuse', label='DT - average')
    plt.legend(handles=[ char_patch, red_patch ])

    ## DATA PLOT

    plt.plot( np.average(out_d,  axis=0)[offset:]  , color="chartreuse" )
    plt.plot( np.average(out_c,  axis=0)[offset:]  , color="red" )

    plt.show()

def cd_convergence_ex2():


    q = np.array( [
        [[-0.375,0.125,0.25],[0.25,-0.5,0.25],[0.25,0.125,-0.375]],
        [[-0.275,0.025,0.25],[0.45,-0.7,0.25],[0.55,0.225,-0.775]],
        [[-0.5,0.25,0.25],[0.11,-0.44,0.33],[0.65,0.42,-1.07]],
        [[-0.3,0.15,0.15],[0.05,-0.5,0.45],[0.35,0.025,-0.375]],
        [[-0.525,0.5,0.025],[0.025,-0.725,0.7],[0.5,0.015,-0.515]]
    ])
    b = np.array( [
        [[0.8,0.05,0.15],[0.05,0.9,0.05],[0.2,0.05,0.75]],
        [[0.7,0.15,0.15],[0.05,0.8,0.15],[0.0,0.05,0.95]],
        [[0.3,0.35,0.35],[0.25,0.7,0.05],[0.2,0.15,0.65]],
        [[0.5,0.5,0.0],[0.3,0.6,0.1],[0.2,0.1,0.7]],
        [[0.2,0.05,0.75],[0.8,0.05,0.15],[0.05,0.9,0.05]],
    ])
    pi = np.array( [
        [0.6,0,0.4],
        [0.3,0.3,0.4],
        [0.6,0.1,0.3],
        [0.4,0.2,0.4],
        [0.8,0.15,0.05],
    ])

    models = 10
    offset = 1

    ##LEGEND

    fig = plt.figure()
    #fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    #fig.subplots_adjust(top=0.85)
    #ax.set_title('Models Comparison')

    ax.set_xlabel('iterations')
    ax.set_ylabel('performance')

    red = mpatches.Patch(color='red', label='CT - special')
    char = mpatches.Patch(color='chartreuse', label='DT - special')
    fire = mpatches.Patch(color='firebrick', label='CT - random')
    olive = mpatches.Patch(color='olivedrab', label='DT - random')

    plt.legend(handles=[ char, red, olive, fire ])



    for mn in range(models):

        print("mn", mn )
        out_c = []
        out_d = []

        if mn < 5:
            Q = q[mn]
            B = b[mn]
            Pi = pi[mn]
            chmm = hmms.CtHMM( Q,B,Pi )
        else:
            chmm = hmms.CtHMM.random( 3,3 , method='exp')

        #hmms.print_parameters( chmm )

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


        #We will train them at our dataset. (It can take a while.)

        #iter_num = 50
        #outd = dt.baum_welch( e,   iter_num, est=True )
        #outc = ct.baum_welch( t,e, iter_num, est=True )

        hidden_states = 3
        runs = 10 #20
        iterations = 150
        out_dt, out_ct = hmms.multi_train_ctdt( hidden_states , t, e, runs, iterations, ret='all', method='unif')

        for ( m, a ) in out_ct:
            out_c.append(a/dreal)


        for ( m, a ) in out_dt:
            out_d.append(a/dreal)

        ## DATA PLOT
        #for i in range(runs):
        #    plt.plot( out_d[i][offset:]  , color="chartreuse" )
        #    plt.plot( out_c[i][offset:]  , color="red" )

        if mn < 5:
            plt.plot( np.average(out_d,  axis=0)[offset:]  , color="chartreuse" )
            plt.plot( np.average(out_c,  axis=0)[offset:]  , color="red" )
        else:
            plt.plot( np.average(out_d,  axis=0)[offset:]  , color="olivedrab" )
            plt.plot( np.average(out_c,  axis=0)[offset:]  , color="firebrick" )


    print("out_c")
    print(out_c)
    print("out_d")
    print(out_d)
    #We can plot and compare both convergence rates. From the essence of models, the continuous model will probably converge a bit slower, but finally will reach the similar value.


    plt.show()

def complexity_ex1():

    #for i in range(2,39,2):
    #    make_time_test_ct( i, 10, 10, 10, 10, True)

    print("@@@")

    #for i in range(2,39,2):
    #    make_time_test_ct( i, 10, 10, 10, 10, False)

    print("@@@")

    #for i in range(2,29,2):
    #    make_time_test_ct( i, 10, 10, 100, 10, True)

    print("@@@")

    for i in range(28,29,2):
        make_time_test_ct( i, 10, 10, 100, 10, False)

def complexity_ex2():

    #for i in [ 2**x for x in range(1,64)]:
    #    make_time_test_ct_od( 10, 10, 10, 10, 10, False, i)

    for i in [ 2**x for x in range(1,64)]:
        make_time_test_ct_od( 10, 10, 10, 10, 10, True, i)

def complexity_ex3():

    for i in [ 2.0**x for x in range(1,256)]:
        make_time_test_ct_od( 5, 5, 5, 10, 5, False, i)


def complexity_ex4():

    time = numpy.zeros( 50 );

    rep = 1

    for j in range(rep):
        print(j)
        for i,val in enumerate( [ 2.0**x for x in range(1,51)] ):

            timex = make_time_test_ct_od2( 5, 5, 5, 10, 5, False, val)
            #print(timex)
            time[i] += timex

    print( "*" * 10 )
    print( time/rep )

def run_precision( n, m, num, data_len, it_num ):
    """Time test, parameters: n-> hidden states, m-> output symbols, num-> number of data vectors, data_len -> length of data vectors,
                              it_num -> number of iterations for Baum-welch algorithm
    """

    chmm = hmms.CtHMM.random( n,m )

    t,e = chmm.generate_data( ( num, data_len ), 0.001 )

    chmm_i = hmms.CtHMM.random( n,m )
    chmm_f = hmms.CtHMM( * chmm_i.params )

    graph_i = chmm_i.baum_welch( t, e, it_num, est=True, method="soft", fast=True )
    graph_f = chmm_f.baum_welch( t, e, it_num, est=True, method="soft", fast=False )

    return (graph_i, graph_f)

def precision_ex():

    it_num = 30

    avg = np.zeros( it_num+1 )

    runs = 10

    for i in range( runs ):
        print(i)
        gi, gf = run_precision( 5, 5, 10, 10, it_num )
        avg = avg + np.fabs(gi-gf)

    avg /= runs

    print(avg)

    plt.plot( range( it_num  ) , avg[:-1] )

    plt.show()


def run_precision2( n, m, num, data_len, it_num ):
    """Time test, parameters: n-> hidden states, m-> output symbols, num-> number of data vectors, data_len -> length of data vectors,
                              it_num -> number of iterations for Baum-welch algorithm
    """

    chmm = hmms.CtHMM.random( n,m )

    t,e = chmm.generate_data( ( num, data_len ), 1 )

    chmm_i = hmms.CtHMM.random( n,m )
    chmm_f = hmms.CtHMM( * chmm_i.params )

    graph_i = chmm_i.baum_welch( t, e, it_num, est=True, method="soft", fast=True )
    graph_f = chmm_f.baum_welch( t, e, it_num, est=True, method="soft", fast=False )

    return (chmm_i.q, chmm_f.q)

def precision_ex2():

    it_num = 30

    avg = np.zeros( it_num )

    runs = 10

    for i in range( runs ):
        print(i)

        for j in range( it_num ):
            A,B = run_precision2( 5, 5, 10, 10, 1 )
            #print(distance(A,B))
            avg[j] = avg[j] + rel_distance(A,B)


    avg /= runs

    print(avg)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('iteration')
    ax.set_ylabel('relative error')

    plt.plot( range( it_num  ) , avg )

    plt.show()

def prob_vit( chmm, t, e ):

    real_v = 0
    for tt,ee in zip(t,e):
       x,_ = chmm.viterbi( tt, ee )
       real_v += x

    return real_v

def prob_vit_orig( chmm1, chmm2, t, e ):

    prob = 0
    for tt,ee in zip(t,e):
       _,seq = chmm2.viterbi( tt, ee )
       prob += chmm1.estimate( seq, tt, ee )

    return prob

def soft_hard2():

    st = 10
    obs = 10

    itr = 50+1

    gs = numpy.zeros( itr+1 )
    gh = numpy.zeros( itr+1 )

    gs2 = numpy.zeros( itr+1 )
    gh2 = numpy.zeros( itr+1 )

    graph_soft = numpy.zeros( itr+1 )
    graph_hard = numpy.zeros( itr+1 )

    g2_soft = numpy.zeros( itr+1 )
    g2_hard = numpy.zeros( itr+1 )


    rvsum = 0
    runs = 5


    for it in range(runs):

        chmm = hmms.CtHMM.random(st,obs)
        t,e = chmm.generate_data( (40,40) )

        print("iter",it)
        chmm_s = hmms.CtHMM.random( st,obs )
        chmm_h = hmms.CtHMM( * chmm_s.params )

        real_v = prob_vit( chmm, t, e )

        for j in range(itr):
            print(j)
            chmm_h.baum_welch( t, e, 1, method="hard" )
            chmm_s.baum_welch( t, e, 1,  method="soft" )

            graph_soft[j] = prob_vit( chmm_s, t, e )
            graph_hard[j] = prob_vit( chmm_h, t, e )

            g2_soft[j] = prob_vit_orig( chmm, chmm_s, t, e )
            g2_hard[j] = prob_vit_orig( chmm, chmm_h, t, e )


        rvsum += real_v
        gs += graph_soft/real_v
        gh += graph_hard/real_v

        gs2 += g2_soft/real_v
        gh2 += g2_hard/real_v


    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('iterations')
    ax.set_ylabel('performance')

    p1, = plt.plot( gs[:-1] / runs, color="firebrick" , label = "soft" )
    p2, = plt.plot( gh[:-1] / runs, color="olivedrab", label = "hard"  )
    p3, = plt.plot( gs2[:-1] / runs, color="red" , label = "soft" )
    p4, = plt.plot( gh2[:-1] / runs, color="chartreuse", label = "hard"  )

    title_proxy = Rectangle((0,0), 0, 0, color='w')
    plt.legend([title_proxy, p1, p2, title_proxy, p3, p4 ], [r'trained model:', "soft","hard",r'original model:', "soft","hard"])


    #plt.legend()

    plt.show()


def soft_hard():
    """ need to remove comments in cthmm for graph2"""

    Q = np.array( [[-0.375,0.125,0.25],[0.25,-0.5,0.25],[0.25,0.125,-0.375]] )
    B = np.array( [[0.8,0.05,0.15],[0.05,0.9,0.05],[0.2,0.05,0.75]] )
    Pi = np.array( [0.6,0,0.4] )

    #chmm = hmms.CtHMM( Q,B,Pi )

    st = 10
    obs = 10

    itr = 100+1

    gs = numpy.zeros( itr+1 )
    gh = numpy.zeros( itr+1 )
    gc = numpy.zeros( itr+1 )

    gs2 = numpy.zeros( itr+1 )
    gh2 = numpy.zeros( itr+1 )
    gc2 = numpy.zeros( itr+1 )
    rsum = 0
    rvsum = 0

    runs = 5

    for it in range(runs):

        print(it)

        chmm = hmms.CtHMM.random(st,obs)
        t,e = chmm.generate_data( (50,10) )

        chmm_s = hmms.CtHMM.random( st,obs )
        chmm_h = hmms.CtHMM( * chmm_s.params )
        chmm_c = hmms.CtHMM( * chmm_s.params )

        print("comb")
        graph_comb, g2_comb = chmm_c.baum_welch( t, e, 5, est=True, method="hard" )
        tmp, tmp2 = chmm_c.baum_welch( t, e, itr-5, est=True, method="soft" )
        g2_comb = np.append( g2_comb[:-1],  tmp2 )
        graph_comb = np.append( graph_comb[:-1],  tmp )
        print("hard")
        graph_hard, g2_hard = chmm_h.baum_welch( t, e, itr, est=True, method="hard" )
        print("soft")
        graph_soft, g2_soft = chmm_s.baum_welch( t, e, itr, est=True, method="soft" )

        real = chmm.data_estimate( t,e )
        real_v = prob_vit( chmm, t, e )

        rsum += real
        rvsum += real_v
        gs += graph_soft/real
        gh += graph_hard/real
        gc += graph_comb/real

        gs2 += g2_soft/real_v
        gh2 += g2_hard/real_v
        gc2 += g2_comb/real_v


    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('iterations')
    ax.set_ylabel('performance')

    #For better visibility of the graph, we cut first two values.
    plt.plot( gs[:-1] / runs, color="red" , label = "soft" )
    plt.plot( gh[:-1] / runs, color="chartreuse", label = "hard"  )
    plt.plot( gc[:-1] / runs, color="olivedrab", label = "comb")
    #plt.rcParams['figure.figsize'] = [20,20]


    plt.legend()

    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('iterations')
    ax.set_ylabel('performance')

    #For better visibility of the graph, we cut first two values.
    plt.plot( gs2[:-1] / runs, color="red", label = "soft"  )
    plt.plot( gh2[:-1] / runs, color="blue", label = "hard"  )
    plt.plot( gc2[:-1] / runs, color="purple", label = "comb" )
    #plt.rcParams['figure.figsize'] = [20,20]

    plt.legend()
    plt.show()

    print( chmm_h.data_estimate( t,e ) )

    hmms.print_parameters( chmm )
    hmms.print_parameters( chmm_s )
    hmms.print_parameters( chmm_h )

    print("prob:")
    print("real", real_v )
    print( "h: ", prob_vit( chmm_h , t, e)   )
    print( "s: ", prob_vit( chmm_s , t, e)   )
    print( "c: ", prob_vit (chmm_c , t, e)   )

    print("distance: ")

    #print( "h: ", dist_vit( chmm, chmm_h , t, e) / real_v  )
    #print( "s: ", dist_vit( chmm, chmm_s , t, e) / real_v  )
    #print( "c: ", dist_vit( chmm, chmm_c , t, e) / real_v  )

def states():

    #Q = np.array( [[-0.511,0.5,0.005,0.005,0.001],[0.001,-0.302,0.15,0.15,0.001],[0.001,0.001,-0.772,0.69,0.08],[0.15,0.001,0.001,-0.302,0.15],[0.001,0.001,0.497,0.001,-0.5]] )
    #B = np.array( [[0.915, 0.035, 0.015, 0.035  ],[0.04, 0.85, 0.075, 0.035 ],[0.035, 0.04, 0.9, 0.025 ],[0.03,0.035,0.035,0.9 ],[0.965, 0.005, 0.005, 0.025 ]] )
    #Pi = np.array( [0.69,0.2,0.05,0.05,0.01] )

    Q = np.array( [[-0.5,0.5,0,0,0],[0.25,-0.5,0.25,0.0,0.0],[0,0.25,-0.5,0.25,0.0,],[0,0,0.25,-0.5,0.25,],[0,0,0,0.5,-0.5]] )
    B = np.array( [[0.85, 0.05, 0.05, 0.05  ],[ 0.05, 0.85, 0.05, 0.05 ],[ 0.05, 0.05,0.85, 0.05 ],[ 0.05, 0.05, 0.05, 0.85],[0.85, 0.05, 0.05, 0.05]] )
    Pi = np.array( [0.4,0.1,0.05,0.05,0.4] )
    Q/=1.5

    chmm = hmms.CtHMM( Q,B,Pi )
    hmms.print_parameters( chmm )

    t,e = chmm.generate_data( (10,10) )
    t2,e2 = chmm.generate_data( (100,100) )
    print(t,e)

    real = chmm.data_estimate( t,e )
    real2 = chmm.data_estimate( t2,e2 )
    itr = 100

    print("start")

    fig = plt.figure(1)
    fig2 = plt.figure(2)

    ax = fig.add_subplot(111)
    ax.set_xlabel('iterations')
    ax.set_ylabel('performance')

    ax = fig2.add_subplot(111)
    ax.set_xlabel('iterations')
    ax.set_ylabel('performance')



    for states in range( 2,9 ):
        chmm2 = hmms.CtHMM.random(states,4)

        train = numpy.zeros(itr)
        test = numpy.zeros(itr)

        for i in range(itr):
            chmm2.baum_welch( t, e, 1 ,method="soft" )
            train[i] = chmm2.data_estimate(t, e)
            test[i] =  chmm2.data_estimate(t2, e2)

        plt.figure(1)
        plt.plot( train[:] / real, label = str(states)  )
        plt.figure(2)
        plt.plot( test[:] / real2, label = str(states)  )

    plt.figure(1)
    plt.legend()
    plt.figure(2)
    plt.legend()

    fig.show()
    fig2.show()
    input()




def main():

    #random.seed(42)

    #make_time_test_dt( 10,10,50,50 ,10 )
    #make_time_test_dt( 4,4,50,150  ,10 )


    ## time experiment watching the complexity of growing states number
    #for i in range(2,21,2):
    #    make_time_test_ct( i, 10, 10, 10, 10)

    #cd_convergence_ex()

    #complexity_ex4()
    #precision_ex()
    #complexity_ex4()
    #precision_ex2()

    #soft_hard2()
    states()


if __name__ == "__main__":
    main()

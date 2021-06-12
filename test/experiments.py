import hmms-custom as hmms
import numpy as np
import time
import numpy
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import scipy.linalg


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

    start_time = time.time()

    hmm.baum_welch( times, data, it_num , fast= fast_val )

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

        dhmm = hmms.DtHMM( *chmm.get_dthmm_params() )
        hmms.print_parameters( dhmm )


        t,_,e = dhmm.generate_data( (50,50), times=True )   # The free space in the return triple is for the state sequences, we do not need them for the training


        creal = chmm.data_estimate(t,e)
        dreal = dhmm.data_estimate(e)
        print("Data estimation by continuous model:", creal)
        print("Data estimation by discrete model:  ", dreal)

        hidden_states = 3
        runs = 10
        iterations =  150
        out_dt, out_ct = hmms.multi_train_ctdt( hidden_states , t, e, runs, iterations, ret='all', method='unif')

        for ( m, a ) in out_ct:
            out_c.append(a/dreal)


        for ( m, a ) in out_dt:
            out_d.append(a/dreal)

    print("out_c")
    print(out_c)
    print("out_d")
    print(out_d)



    ##LEGEND

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.set_xlabel('iterations')
    ax.set_ylabel('performance ratio')

    plt.legend()

    ## DATA PLOT

    for i in range(runs*models):
        plt.plot( out_d[i][offset:]  , label='DT - single run', color = 'darkorange' )
        plt.plot( out_c[i][offset:]  , label='CT - single run', color = 'blue' )



    ##LEGEND

    fig2 = plt.figure()

    ax2 = fig2.add_subplot(111)

    ax2.set_xlabel('iterations')
    ax2.set_ylabel('performance ratio')

    ## DATA PLOT

    plt.plot( np.average(out_d,  axis=0)[offset:]  , label='DT - average', color = 'darkorange' )
    plt.plot( np.average(out_c,  axis=0)[offset:]  , label='CT - average', color = 'blue' )

    plt.legend()

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

    ax = fig.add_subplot(111)

    ax.set_xlabel('iterations')
    ax.set_ylabel('performance ratio')





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


        dhmm = hmms.DtHMM( *chmm.get_dthmm_params() )
        hmms.print_parameters( dhmm )


        t,_,e = dhmm.generate_data( (50,50), times=True )   # The free space in the return triple is for the state sequences, we do not need them for the training


        creal = chmm.data_estimate(t,e)
        dreal = dhmm.data_estimate(e)
        print("Data estimation by continuous model:", creal)
        print("Data estimation by discrete model:  ", dreal)


        hidden_states = 3
        runs = 10 #20
        iterations = 150
        out_dt, out_ct = hmms.multi_train_ctdt( hidden_states , t, e, runs, iterations, ret='all', method='unif')

        for ( m, a ) in out_ct:
            out_c.append(a/dreal)


        for ( m, a ) in out_dt:
            out_d.append(a/dreal)

        ## DATA PLOT


        if mn < 5:
            plt.plot( np.average(out_d,  axis=0)[offset:]  , label='DT - special', color = 'darkorange' )
            plt.plot( np.average(out_c,  axis=0)[offset:]  , label='CT - special',color = 'blue' )
        else:
            plt.plot( np.average(out_d,  axis=0)[offset:]  , label='DT - random',color = 'red' )
            plt.plot( np.average(out_c,  axis=0)[offset:]  , label='CT - random',color = 'darkblue' )




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


def run_precision2( chmm_i, chmm_f, t, e, it_num ):
    """Run both float and integer algorithm baum-welch"""

    graph_i = chmm_i.baum_welch( t, e, it_num, est=True, method="soft", fast=True )
    graph_f = chmm_f.baum_welch( t, e, it_num, est=True, method="soft", fast=False )

def precision_ex2():

    it_num = 50

    avg = np.zeros( it_num + 1)
    avg[0] = 0

    runs = 10

    n = 5
    m = 5
    num = 10
    data_len = 10

    for i in range( runs ):
        print(i)

        chmm = hmms.CtHMM.random( n,m )

        t,e = chmm.generate_data( ( num, data_len ), 0.001 )

        chmm_i = hmms.CtHMM.random( n,m )
        chmm_f = hmms.CtHMM( * chmm_i.params )

        for j in range( 1,it_num+1 ):

            run_precision2(chmm_i, chmm_f, t, e, 1 )

            avg[j] = avg[j] + rel_distance(chmm_i.q,chmm_f.q)


    avg /= runs

    print(avg)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('iteration')
    ax.set_ylabel('relative error')

    plt.plot( range( it_num + 1 ) , avg )

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


    plt.show()

def soft_hard_simple():

    Q = np.array( [[-0.375,0.125,0.25],[0.25,-0.5,0.25],[0.25,0.125,-0.375]] )
    B = np.array( [[0.8,0.05,0.15],[0.05,0.9,0.05],[0.2,0.05,0.75]] )
    Pi = np.array( [0.6,0,0.4] )

    chmm = hmms.CtHMM( Q,B,Pi )

    t,e = chmm.generate_data( (100,100) )

    chmm_s = hmms.CtHMM.random( 3,3 )
    chmm_h = hmms.CtHMM( * chmm_s.params )
    chmm_c = hmms.CtHMM( * chmm_s.params )

    print("comb")
    #graph_comb = chmm_c.baum_welch( t, e, 5, est=True, method="hard" )
    #graph_comb = np.append( graph_comb,  chmm_c.baum_welch( t, e, 95, est=True, method="soft" ) )
    print("hard")
    graph_hard = chmm_h.baum_welch( t, e, 100, est=True, method="hard" )
    print("soft")
    graph_soft = chmm_s.baum_welch( t, e, 100, est=True, method="soft" )


    real = chmm.data_estimate( t,e )


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('iterations')
    ax.set_ylabel('performance ratio')
    #For better visibility of the graph, we cut first two values.
    plt.plot( graph_soft[1:] / real, color = 'blue' , label = "soft method" )
    plt.plot( graph_hard[1:] / real, color = 'darkorange', label = "hard method"   )

    plt.legend()


    ##plt.plot( graph_comb[1:-1] / real, color="purple")

    plt.show()


def baum_welch_test( chmm, t, e, tt, et, _method, itr ):
    """ return convergence line on both training (t,e) and testing datasets (tt,et) for dataset likelihood and viterbi"""

    train = numpy.zeros( itr+1 )
    test = numpy.zeros(  itr+1 )

    for i in range(itr):
        print(i+1,'/',itr)

        train[i] = chmm.data_estimate(t, e)
        test[i] =  chmm.data_estimate(tt, et)

        chmm.baum_welch( t, e, 1 , est=False ,method=_method )

    train[-1] = chmm.data_estimate(t, e)
    test[-1] =  chmm.data_estimate(tt, et)

    return (train,test)


def bw_test( chmm, t, e, tt, et, _method, itr ):
    """ return convergence line on both training (t,e) and testing datasets (tt,et) for dataset likelihood and viterbi"""

    train = numpy.zeros( itr+1 )
    test = numpy.zeros(  itr+1 )

    train_vit = numpy.zeros( itr+1 )
    test_vit = numpy.zeros(  itr+1 )

    for i in range(itr):

        train[i] = chmm.data_estimate(t, e)
        test[i] =  chmm.data_estimate(tt, et)
        train_vit[i] = prob_vit(chmm, t, e)
        test_vit[i] =  prob_vit(chmm, tt, et)
        chmm.baum_welch( t, e, 1 , est=False ,method=_method )

    train[-1] = chmm.data_estimate(t, e)
    test[-1] =  chmm.data_estimate(tt, et)
    train_vit[-1] = prob_vit(chmm, t, e)
    test_vit[-1] =  prob_vit(chmm, tt, et)

    return [train,test,train_vit,test_vit]


def soft_hard3():
    """ need to remove comments in cthmm for graph2"""

    Q = np.array( [[-0.375,0.125,0.25],[0.25,-0.5,0.25],[0.25,0.125,-0.375]] )
    B = np.array( [[0.8,0.05,0.15],[0.05,0.9,0.05],[0.2,0.05,0.75]] )
    Pi = np.array( [0.6,0,0.4] )

    chmm = hmms.CtHMM( Q,B,Pi )

    st = 3
    obs =3
    #chmm = hmms.CtHMM.random( st, obs )


    itr = 100+1

    s = numpy.zeros( (4,itr+1) )
    h = numpy.zeros( (4,itr+1) )

    rsum = numpy.zeros( 4 )


    runs = 1

    for it in range(runs):

        print(it)

        #chmm = hmms.CtHMM.random(st,obs)
        t,e = chmm.generate_data( (25,25) )
        tt,et = chmm.generate_data( (2,2) ) #test dataset

        rsum[0] = chmm.data_estimate( t,e )
        rsum[2] = prob_vit( chmm, t, e )

        rsum[1] = chmm.data_estimate( tt,et )
        rsum[3] = prob_vit( chmm, tt, et )

        chmm_s = hmms.CtHMM.random( st,obs )
        chmm_h = hmms.CtHMM( * chmm_s.params )

        s_temp = bw_test(chmm_s,t,e,tt,et,"soft",itr)
        h_temp = bw_test(chmm_h,t,e,tt,et,"hard",itr)

        for i in range(4):
            s[i] += s_temp[i] / rsum[i]
            h[i] += h_temp[i] / rsum[i]

    for i in range( 4 ):
        fig = plt.figure(i)
        ax = fig.add_subplot(111)
        ax.set_xlabel('iterations')
        ax.set_ylabel('performance')
        plt.figure(i)
        plt.plot( s[i] / runs, label = "soft method"  )
        plt.plot( h[i] / runs, label = "hard method"  )
        plt.legend()
        fig.show()

    input()





def soft_hard():
    """ need to remove comments in cthmm for graph2"""

    Q = np.array( [[-0.375,0.125,0.25],[0.25,-0.5,0.25],[0.25,0.125,-0.375]] )
    B = np.array( [[0.8,0.05,0.15],[0.05,0.9,0.05],[0.2,0.05,0.75]] )
    Pi = np.array( [0.6,0,0.4] )

    #chmm = hmms.CtHMM( Q,B,Pi )

    st = 2
    obs = 2

    itr = 10+1

    gs = numpy.zeros( itr+1 )
    gh = numpy.zeros( itr+1 )
    gc = numpy.zeros( itr+1 )

    gs2 = numpy.zeros( itr+1 )
    gh2 = numpy.zeros( itr+1 )
    gc2 = numpy.zeros( itr+1 )
    rsum = 0
    rvsum = 0

    runs = 2

    for it in range(runs):

        print(it)

        chmm = hmms.CtHMM.random(st,obs)
        t,e = chmm.generate_data( (5,5) )
        tt,et = chmm.generate_data( (10,5) ) #test dataset

        chmm_s = hmms.CtHMM.random( st,obs )
        chmm_h = hmms.CtHMM( * chmm_s.params )
        chmm_c = hmms.CtHMM( * chmm_s.params )

        """print("comb")
        graph_comb, g2_comb = chmm_c.baum_welch( t, e, 5, est=True, method="hard" )
        tmp, tmp2 = chmm_c.baum_welch( t, e, itr-5, est=True, method="soft" )
        g2_comb = np.append( g2_comb[:-1],  tmp2 )
        graph_comb = np.append( graph_comb[:-1],  tmp )"""
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
    B = np.array( [[0.85, 0.09, 0.05, 0.01  ],[ 0.07, 0.85, 0.07, 0.01 ],[ 0.01, 0.07,0.85, 0.07 ],[ 0.07, 0.01, 0.07, 0.85],[0.85, 0.01, 0.05, 0.09]] )
    Pi = np.array( [0.4,0.1,0.05,0.05,0.4] )
    Q/=2

    chmm = hmms.CtHMM( Q,B,Pi )
    hmms.print_parameters( chmm )

    #big dataset
    t,e = chmm.generate_data( (100,100) )
    #small dataset
    #t,e = chmm.generate_data( (15,15) )

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

    train = numpy.zeros( (9,itr+1) )
    test = numpy.zeros( (9,itr+1) )

    runs = 5

    for run in range(runs):

        print("run", run)

        for states in range( 2,9 ):

            print("states", states)

            chmm2 = hmms.CtHMM.random(states,4)

            for i in range(itr):

                train[states,i] += chmm2.data_estimate(t, e) / real
                test[states,i] +=  chmm2.data_estimate(t2, e2) / real2
                chmm2.baum_welch( t, e, 1 ,method="soft" )

            train[states,-1] += chmm2.data_estimate(t, e) / real
            test[states,-1] +=  chmm2.data_estimate(t2, e2) / real2


    print("train",train)
    print("test",test)

    for states in range(2,9):

        plt.figure(1)
        plt.plot( train[states] / runs, label = str(states)  )
        plt.figure(2)
        plt.plot( test[states] / runs, label = str(states)  )

    plt.figure(1)
    plt.legend()
    plt.figure(2)
    plt.legend()

    fig.show()
    fig2.show()
    input()


def random_3diag( n, all_rand ):

    mask = np.eye(n, k=1) + np.eye(n,k=0) + np.eye(n,k=-1)

    Q = np.random.rand(n,n)
    Q *= mask

    if all_rand:
        B = np.random.rand(n,n)
        Pi = np.random.rand(n)

    else:
        B = np.eye(n, k=1)*0.05 + np.eye(n,k=0)*0.9 + np.eye(n,k=-1)*0.05
        Pi= np.ones(n)

    B *= mask

    for i in range(n):
        B[i] /= np.sum( B[i] )
        Q[i,i] = - np.sum( Q[i] ) + Q[i,i]
    Pi /= np.sum( Pi )

    return hmms.CtHMM( Q,B,Pi )

def states3():

    s = []
    d = []

    states = 30
    iteration = 100


    model = random_3diag( states , False )
    sparse = random_3diag( states , True )
    dense =  hmms.CtHMM.random(states,states)

    t,e = model.generate_data( (100,100) )
    tt,et = model.generate_data( (100,100) )

    real = model.data_estimate( t,e )
    real_t = model.data_estimate( tt,et )



    s_train, s_test = baum_welch_test( sparse, t, e, tt, et,"soft", iteration )
    d_train, d_test = baum_welch_test( dense,  t, e, tt, et, "soft", iteration  )

    print(repr(s_train/real))
    print(repr(s_test/real_t))
    print(repr(d_train/real))
    print(repr(d_test/real_t))


def states2():

    s = []
    d = []

    rng = range(275 ,280,5)

    for states in rng:

        model = random_3diag( states , False )
        sparse = random_3diag( states , True )
        dense =  hmms.CtHMM.random(states,states)

        t,e = model.generate_data( (10,100) )

        #print(t,e)

        start_time = time.time()
        sparse.baum_welch( t, e, 1 ,method="soft" )
        stime = time.time() - start_time

        s.append( stime )
        print(states, " states -> time:" , stime )

        if(states <= 75 ):

            start_time = time.time()
            dense.baum_welch( t, e, 1 ,method="soft" )
            dtime = time.time() - start_time
            d.append( dtime )
            print(states, " states -> time:" , dtime )

        print(s)
        print(d)

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.set_xlabel('number of states')
    ax.set_ylabel('time [s]')

    plt.plot( rng, s, label = "sparse matrix"  )
    plt.plot( rng, d, label = "dense matrix"  )
    plt.legend()
    fig.show()
    input()


def expm_time( val ):

    shape = (100,100)

    Q = numpy.full( shape, val ) - shape[0] * numpy.eye( shape[0] ) * val

    start_time = time.time()

    Q2 = scipy.linalg.expm( Q )

    ret = (time.time() - start_time)

    print(Q2)

    return ret


def expm_test():

    time = numpy.zeros( 50 );

    rep = 5

    for j in range(rep):
        print(j)
        for i,val in enumerate( [ 2.0**x for x in range(1,51)] ):

            timex = expm_time( (val + random.randint(1, 100))*0.888978745 )
            #print(timex)
            time[i] += timex

    print( "*" * 10 )
    print( time/rep )

    X = [ 2**x for x in range(1,51)]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('approximate length of the time intervals')
    ax.set_ylabel('time [s]')

    ax.set_xscale("log", nonposx='clip')


    plt.plot(X, time/rep )  #whole time -al:float

    plt.legend()

    plt.show()

def empty():

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('most probable sequence by the original model')
    ax.set_ylabel('performance')

    fig.show()

    input()

def mle():
    """TODO move it to the tests"""

    data_l = [ np.array( [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] ) ,
         np.array( [0, 1, 0, 0, 1, 0, 1 ] ),
         np.array( [2, 0, 1, 0, 2, 0, 0, 0, 0, 0] ) ]

    data_n = np.array(  [[0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 2, 2, 2, 1, 0, 1, 0],
        [2, 0, 1, 0, 0, 0, 0, 0, 0, 0]] )


    dhmm1 = hmms.DtHMM.random( 3,3 )
    dhmm2 = hmms.DtHMM.random( 3,3 )


    hmms.print_parameters(dhmm1)
    hmms.print_parameters(dhmm2)

    dhmm2.maximum_likelihood_estimation( data_l, data_l )

    hmms.print_parameters(dhmm2)

def mle_c2():

    s = [ np.array( [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] ) ,
         np.array( [0, 1, 0, 0, 1, 0, 1 ] ),
         np.array( [2, 0, 1, 0, 2, 0, 0, 0, 0, 0] ) ]
    e = [ np.array( [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1] ) ,
         np.array( [0, 1, 0, 0, 1, 0, 1 ] ),
         np.array( [2, 0, 1, 0, 2, 0, 0, 0, 0, 0] ) ]
    t = [ np.array( [0, 2, 4, 5, 8, 11, 14, 16, 17, 19, 21, 23] ) ,
         np.array( [0, 1, 4, 6, 9, 10, 11 ] ),
         np.array( [0, 4, 8, 10, 12, 16, 17, 18, 19, 20] ) ]

    chmm = hmms.CtHMM.random(3,3)
    print( "e1", chmm.full_data_estimate(s,t,e) )
    chmm.maximum_likelihood_estimation(s,t,e)

    print( "e2", chmm.full_data_estimate(s,t,e) )


def mle_c():

    chmm_g = hmms.CtHMM.random(3,3)
    t,s,e = chmm_g.generate_data( ( 100, 100 ), states=True )

    real = chmm_g.full_data_estimate(s,t,e)
    print( "real", real )

    chmm = hmms.CtHMM.random(3,3)

    graph = chmm.maximum_likelihood_estimation(s,t,e,100,est=True)
    print( graph )

    plt.plot( graph / real )
    plt.show()
    input()


def main():

    #random.seed(42)

    #make_time_test_dt( 10,10,50,50 ,10 )
    #make_time_test_dt( 4,4,50,150  ,10 )


    ## time experiment watching the complexity of growing states number
    #for i in range(2,21,2):
    #    make_time_test_ct( i, 10, 10, 10, 10)

    #cd_convergence_ex1()

    #complexity_ex4()
    #precision_ex()
    #complexity_ex4()
    #precision_ex2()

    #soft_hard_simple()
    #soft_hard3()
    #states2()
    #expm_test()

    #empty()

    #states3()

    mle_c()


if __name__ == "__main__":
    main()

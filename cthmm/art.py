import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import pandas as pd
from IPython.display import display

def print_parameters( hmm ):

    print("Initial probabilities (π) :")
    display( pd.DataFrame(hmm.pi) )
    print("Transition probabilities matrix (A):")
    display( pd.DataFrame(hmm.a) )
    print("Emission probabilities matrix (B):")
    display( pd.DataFrame(hmm.b) )

def plot_hmm( s_seq, e_seq ):

    n = e_seq.shape[0]
    X = np.arange(n)
    Y0 = np.zeros(n)
    Y1 = np.ones(n)

    fig, ax = plt.subplots()

    ax.set_aspect('equal')

    plt.xlim( np.amin(X)-1, np.amax(X)+1 ), plt.xticks([])
    plt.ylim(-2,3), plt.yticks([])

    e_num = np.amax( e_seq )+1
    s_num = np.amax( s_seq )+1

    for (x,y,c) in zip(X,Y1,s_seq):
        plt.annotate( c , xy=(x, y), xycoords='data', xytext=(-5, -5), textcoords='offset points', fontsize=16 )
        ax.add_artist(plt.Circle((x, y), 0.3, color=cm.gnuplot( c/s_num ), alpha=0.4))
        ax.arrow(x-0.7, y, 0.3, 0, head_width=0.35, head_length=0.1, fc='k', ec='k')
        ax.arrow(x, y-0.3, 0, -0.3, head_width=0.35, head_length=0.1, fc='k', ec='k')

    for (x,y,c) in zip(X,Y0,e_seq):
        ax.add_artist(plt.Circle((x, y), 0.3, color=cm.gnuplot( 0.9*c/e_num + 0.1), alpha=0.7))
        plt.annotate( c , xy=(x, y), xycoords='data', xytext=(-5, -5), textcoords='offset points', fontsize=16 )

    plt.show()


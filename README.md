**UPDATE 2023/Feb/27** Direct Pypi installation is now fixed.

Intro
=======

HMMs is the **Hidden Markov Models library** for *Python*. 
It is easy to use **general purpose** library implementing all the important
submethods needed for the training, examining and experimenting with
the data models.

The computationally expensive parts are powered by
*Cython* to ensure high speed.

The library supports the building of two models:

<dl>
 <dt>Discrete-time Hidden Markov Model</dt>
 <dd>Usually simply referred to as the Hidden Markov Model.</dd>

 <dt>Continuous-time Hidden Markov Model</dt> 
 <dd>The variant of the Hidden Markov Model where the state transition as well as observations occurs in the continuous time. </dd>
</dl>

Before starting work, you may check out **the tutorial with examples**. [the ipython notebook](https://github.com/lopatovsky/CT-HMM/blob/master/hmms.ipynb), covering most of the common use-cases.

For **the deeper understanding** of the topic refer to the corresponding [diploma thesis](https://github.com/lopatovsky/DP).
Or read some of the main referenced articles: [Dt-HMM](http://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf), [Ct-HMM](https://web.engr.oregonstate.edu/~lif/nips2015_CTHMM_learning_camera_ready.pdf) .

-  Sources of the project:
   [Pypi](https://pypi.python.org/pypi/hmms),
   [Github](https://github.com/lopatovsky/CT-HMM),


Requirements
-------------

-  python 3.5
-  libraries: Cython, ipython, matplotlib, notebook, numpy, pandas, scipy,
-  libraries for testing environment: pytest   

Download & Install
-------------------

The Numpy and Cython must be installed before installing the library package from pypi.

```
(env)$ python -m pip install numpy cython
(env)$ python -m pip install hmms

```



Intro
=======

HMMs is the **Hidden Markov Models library** for *Python*. 
It is easy to use **general purpose** library, implementing all the important submethods, what you can need for the training, examining and experimenting with the data models.

The effectivness of the computationally expensive parts is powered by *Cython*.

You can build two models:

**Discrete-time Hidden Markov Model** 
  
  Usually just reffered as the Hidden Markov Model.

**Continuous-time Hidden Markov Model** 
  
  The variant of the Hidden Markov Model, where the state transition can occure in the continuous time, and that allows random distribution  of the observation times.

For the **fast** understanding of the models theory read the `Simple guide to HMMs <https://example.com/#TODO>`__,

Before starting to work, it is recommended to go trough **tutorial with examples** `the ipython notebook <https://github.com/lopatovsky/CT-HMM/blob/master/CT-HMM.ipynb>`__, covering most of the main usecases.

For **deeper understanding** of the topic you can see the corresponding `diploma thesis <https://github.com/lopatovsky/DP>`__.
Or read the main referenced articles: `Dt-HMM <http://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf>`__, `Ct-HMM <https://web.engr.oregonstate.edu/~lif/nips2015_CTHMM_learning_camera_ready.pdf>`__

-  Sources of the project:
   `Testpypi <https://testpypi.python.org/pypi?%3Aaction=pkg_edit&name=hmms>`__,
   `Github <https://github.com/lopatovsky/CT-HMM>`__,
   `Read The Docs <https://example.com/#TODO>`__.

Requirements
~~~~~~~~~~~~

-  python 3.5
-  libraries: Cython, ipython, matplotlib, notebook, numpy, pandas, scipy,
-  libraries for testing environment: pytest   

Download & Install
~~~~~~~~~~~~~~~~~~

You can install the package directly from test pypi by command:

``$ python -m pip install --extra-index-url https://testpypi.python.org/pypi hmms``



Intro
=======

The Greattwitterwall is the Twitter based application, that anable the search
of the key-words in most recent tweets.

The project rised as the part of the homework serie for MI-PYT at FIT CTU in Prague.

The Program operates in two modes:

-  The Console Application:

Anable to search for the tweets, refreshing in the realtime, when new tweet with the searching keyword arise.  

-  The Web Application:  

Use the web frontend to format searched tweets in more readible way.

-  Sources:
   `Testpypi <https://testpypi.python.org/pypi?%3Aaction=pkg_edit&name=greattwitterwall>`__,
   `Github <https://github.com/lopatovsky/greattwitterwall>`__,
   `Read The Docs <https://readthedocs.org/projects/greattwitterwall/>`__.

If you only want to use it for searching tweets or see its functionality, you can find it `here <http://jamaisvu.pythonanywhere.com/MI-PYT/>`__. 

Feel free to mend the url in any form to search for desired key-words.

``http://jamaisvu.pythonanywhere.com/{searched word}/`` 

Requirements
~~~~~~~~~~~~

-  python 3.5
-  libraries: `click <http://click.pocoo.org/6/>`__,
   `flask <http://flask.pocoo.org/>`__,
   `requests <http://docs.python-requests.org/en/master/>`__,
   `jinja2 <http://jinja.pocoo.org/docs/dev/>`__,   


.. _api-secret-key:

Download & Install
~~~~~~~~~~~~~~~~~~

You can install the package directly from test pypi by command:

``$ python -m pip install --extra-index-url https://testpypi.python.org/pypi greattwitterwall``

Or download at `Github <https://github.com/lopatovsky/greattwitterwall>`__ and install from source:

``$ python setup.py install``

To succesfully run the program, you need to create the file "auth.cfg" in the main directory in the form ::

   [twitter]
   key = XXXX
   secret = YYYY

You can get twitter key and secret by registering the app `here <https://apps.twitter.com/>`__.

Usage
~~~~~

You may launch the console or web application by command: 

``$ greattwitterwall console``

``$ greattwitterwall web``

Then just follow the prompt questions or
use the ``--help`` option to get more information.

delete me ref :ref:`api-secret-key`

Examples
~~~~~~~~

.. testsetup::

   import greattwitterwall as gtw

.. doctest::

   >>> gtw.suma(5,6)
   11
   
   >>> gtw.suma(-4,3)
   -1


Other examples may be added in future.





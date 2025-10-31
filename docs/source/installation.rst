.. _install: 

Installation
============

Requirements
------------

You will need a working Python 3.10+ installation.

Install from source
-------------------

Alternatively, you can install the latest version directly from the most up-to-date version
of the source-code by cloning/forking the GitHub repository 

.. code-block:: bash

    git clone https://github.com/mikegrudic/pism.git


Once you have the source, you can build pism (and add it to your environment)
by executing

.. code-block:: bash

    pip install .

in the top level directory. The required Python packages will automatically be 
installed as well.

You can test your installation by importing the pism Python frontend in Python:

.. code-block:: python

    import pism

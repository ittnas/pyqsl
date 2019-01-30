Python framework for simulating quantum mechanics - pyqsl
=========================================================
pyqsl provides tools to run simulations and save data in a few different formats. The core functionality is to provide framework for sweeping simulation parameters.

The simulation data can be saved and viewed using Labber LogBrowser (http://labber.org/).

Installation
------------
pyqsl can be installed using pip, but is not yet available through public repositories. To install, first clone the repository and enter the directory. There run ::

  pip3 install -r requirements.txt .

Optionally, can be installed installed in a virtual environment using virtualenv. In order to create and activate the virtual environment, run ::
  
  python3 -m venv env
  source env/bin/activate
  pip3 install -r requirements.txt .

To verify that the installation works there is a directory containing tests.

For automated document generation the library uses Sphinx. To generate the documentation install sphinx ::

  pip3 install sphinx

And then navigate to doc/ directory. To update the API doc strings, run ::

  sphinx-apidoc -f -o source/ ../pyqsl

and then build the docs with ::

  make html

to generate html documentation.

TODO
----


Issues
------
* Labber path is hardcoded in core.py. This should be changed.
  


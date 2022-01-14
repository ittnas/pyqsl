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

to generate the html documentation. The documentation can be found in doc/build/html.

TODO
----


Issues
------
* Labber path is hardcoded in core.py. This should be changed.

* There is a weird feature related to multiprocessing in windows. Every time a new process is spawned, the parent class is imported. This leads to an infinite loop if care is not taken. Therefore, every time pyqsl.core.simulation_loop is called, a main guard needs to be added, e.g.
```python
if __name__=='__main__':
    result = pyqsl.core.simulation_loop(p.__dict__, simulation_task, sweep_arrays=sweep_arrays, expand_data=True, parallelize=True)

```
  


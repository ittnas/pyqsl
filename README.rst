Python framework for simulating quantum mechanics - pyqsl
=========================================================
pyqsl provides a framework for running tasks as a function of different combinations of tasks input arguments. For example, pyqsl supports sweeping over
any combination of the parameters, expanding the results of the task to numpy arrays of correct dimensions. Additinoanally, pyqsl provides a way to
define relations between different task-parameters in order to construct complicated sweep spaces. The relations can be arbitrarily nested, and support
different formats such as symbolic equations and lookuptables.

Installation
------------
pyqsl can be installed using pip, but is not yet available through public repositories. To install, first clone the repository and enter the directory. There run ::

  git clone https://github.com/ittnas/pyqsl.git
  pip install ./pyqsl

For development, install also the tools needed for testing ::

  pip install -e ./pyqsl[testing]

For automated document generation the library uses Sphinx. To generate the documentation install sphinx and then navigate to doc/ directory. To update the API doc strings, run ::

  sphinx-apidoc -f -o source/ ../pyqsl

and then build the docs with ::

  make html

to generate the html documentation. The documentation can be found in doc/build/html.


Issues
------
* There is a weird feature related to multiprocessing in windows. Every time a new process is spawned, the parent class is imported. This leads to an infinite loop if care is not taken. Therefore, every time pyqsl.core.simulation_loop is called, a main guard needs to be added, e.g.
```python
if __name__=='__main__':
    result = pyqsl.run(simulation_task, settings, parallelize=True)
```
  


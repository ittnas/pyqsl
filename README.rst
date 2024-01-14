Python framework for simulating quantum mechanics - pyqsl
=========================================================
pyqsl provides a framework for running tasks as a function of different combinations of tasks input arguments. For example, pyqsl supports sweeping over
any combination of the parameters, expanding the results of the task to numpy arrays of correct dimensions. Additionally, pyqsl provides a way to
define relations between different task-parameters in order to construct complicated sweep spaces. The relations can be arbitrarily nested, and support
different formats such as symbolic equations and lookuptables.

Installation
------------
pyqsl is available through PyPI and can be installed with pip. ::

  pip intall pyqsl
  
For manual installation, first clone the repository and enter the directory. There run ::

  git clone https://github.com/ittnas/pyqsl.git
  pip install ./pyqsl

For development, install also the tools needed for testing ::

  pip install -e ./pyqsl[testing]

For running the examples, the additional requirements can be installed as ::

  pip install pyqsl[examples]
  
For automated document generation the library uses Sphinx. To generate the documentation install sphinx and then navigate to doc/ directory. To update the API doc strings, run ::

  sphinx-apidoc -f -o source/ ../

and then build the docs with ::

  make html

to generate the html documentation. The documentation can be found in doc/build/html.

Usage example
-------------
The following simple example demonstrates how pyqsl can be used to run tasks and sweep over model parameters. Let's first create a simple cosine function which values we want to evalaute.

.. code-block:: python

    import pyqsl
    import numpy as np
    import matplotlib.pyplot as plt
    
    def cosine(amplitude, phase, frequency, time):
        return amplitude*np.cos(2*np.pi*(time*frequency + phase))
    
    settings = pyqsl.Settings()
    settings.amplitude = 2
    settings.phase = np.pi
    settings.frequency = pyqsl.Setting(relation='2*amplitude', unit='Hz')
    settings.time = pyqsl.Setting(unit='s')
    sweeps = {'amplitude': np.linspace(0, 1, 101), 'time': np.linspace(0, 5, 101)}
    result = pyqsl.run(cosine, settings=settings, sweeps=sweeps)
    result.dataset.data.plot()
    plt.show()

This above calculates the value of the cosine function when the input parameters ``amplitude`` and ``time`` are varied. Additionally, there is a relation set for frequency, which sets its value to depend on amplitude so that cosine oscillates faster for higher amplitude values. Finally, the result is plotted using ``result.dataset.data.plot()``.

Issues
------
* There is a weird feature related to multiprocessing in windows. Every time a new process is spawned, the parent class is imported. This leads to an infinite loop if care is not taken. Therefore, every time pyqsl.core.simulation_loop is called, a main guard needs to be added, e.g. ::

    if __name__=='__main__':
        result = pyqsl.run(simulation_task, settings, parallelize=True)

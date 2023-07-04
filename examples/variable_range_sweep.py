"""This example demonstrates how to perform a variable range sweep of a target function"""
##
import numpy as np
import pyqsl
import matplotlib.pyplot as plt

# Define a target function here


def height(x, y):
    return {'z': (np.cos(2*np.pi*x) - y)**2}

# Construct settings with some defaults


settings = pyqsl.Settings()
settings.x = 0
settings.x.unit = 'm'
settings.y = 0
settings.y.unit = 'm'
settings.z = 0
settings.z.unit = 'm'
# Instead of sweeping in a rectangle for x and y, we want to sweep around the minimum of the target funtion.
# Construct a relation that sweeps y around a cosine function of x.
# Notice that this is not a circular relation. The value of y in the equation refers to original value of y.
# The result of the relation will create an "evaluated value", which will be used in the final calculation.
# The original value of y will be adjusted by the sweep.
settings.y.relation = pyqsl.Equation(equation='cos(2*3.1415*x) + y')

sweeps = {'x': np.linspace(-1, 1, 501), 'y': np.linspace(-1, 1, 501)}
result = pyqsl.run(height, settings=settings, sweeps=sweeps, parallelize=False)
plt.pcolor(result.x, result.y, result.z, cmap='Greens')
plt.xlabel(f'x ({settings.x.unit})')
plt.ylabel(f'y ({settings.y.unit})')
plt.show()
result.save('test_save.data')
##
sweeps = {'x': np.linspace(-1, 1, 501)}
result = pyqsl.run(height, settings=settings, sweeps=sweeps, parallelize=True)
plt.plot(result.x, result.z)
plt.plot(result.y, result.z)
plt.show()

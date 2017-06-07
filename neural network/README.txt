Neural network with backpropagation and metric-tensor learning, written
in python.

I started this project mainly to learn Python, and additionally, to test
an idea about the geometry of neural networks.

NetworkClasses.py - neural network in python
NumpyNetworkClasses.p - neural network using numpy
MetricNetworkClasses.py - neural network using metric-tensor propagation
Main.py - runs the neural network with two test cases
MainNumpy.py - the same for the numpy version
MainMetric.py - the same for the metric-tensor version

The MetricNetworkClasses learning method creates a metric-tensor by
treating the weights as a complicated coordinate system in the space
of output nodes.  The one-form defined by the difference between the
desired and actual output is pulled back to weight space (equivalent
to the gradient of the error squared), then converted to a vector 
with the metric tensor.


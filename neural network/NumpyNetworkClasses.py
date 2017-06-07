#Simple neural network with backpropagation learning
#Created by Mark Palenik
#A replacement for NetworkClasses that does the same thing, but
#using numpy, and possibly some slightly better python

import numpy as np
import random
import math

narray = np.random.random([2,1])
print(narray.shape[0])

class Node:
    def __init__(self):
        self.weights = [] #weights connecting to next layer
        self.invalue = 0  #input from previous layer
        self.outvalue = 0 #output to next layer
        self.olddws = [] #for momentum

class Layer:
    def __init__(self):
        self.Nodes = []
        self.nnodes = 0

class Network:
    def __init__(self):
        self.alpha = 0.1 #momentum
        self.eta = 10.0   #scale factor on gradient
        self.Nlayers = 0
        self.Layers = []
        
    def ActivationFunction(self,x):
        #activation function, bound between 0 and 1
        return 1.0/(1.0+math.exp(-x))
    
    def ActivationDeriv(self,x):
        #derivative of activation function
        denom = (1.0+math.exp(-x))
        return math.exp(-x)/(denom*denom)
        
    def AddLayers(self,nNodes):
        #Add layers to neural network.  nNodes is a list
        #containing the number of nodes in each layer
        
        nAdd = len(nNodes)
        
        for i in range(nAdd):
            L = Layer()
            L.nnodes=nNodes[i]
            for j in range(nNodes[i]): L.Nodes.append(Node())
            self.Layers.append(L)
            
            if (self.Nlayers>0):
                for node in self.Layers[self.Nlayers-1].Nodes:
                    node.olddws = np.zeros(nNodes[i],dtype=np.double)
                    node.weights = 2.0*(np.random.random(nNodes[i]) - 0.5*np.ones(nNodes[i],dtype=np.double))
            
            self.Nlayers = self.Nlayers+1
            
    def AddLayer(self,nNodes):
        #wrapper for AddLayers to add a single layer with nNodes nodes
        AddLayers(1,[nNodes])
            
    def RunNetwork(self,input):
        #forward propagate the network.  input is a list containing the
        #input.  Returns a list with the output from each node in the output layer
        
        currLayer = self.Layers[0]
        for i in range(currLayer.nnodes): currLayer.Nodes[i].outvalue = input[i]
        LastLayer = currLayer
        
        for i in range(1,self.Nlayers):
            currLayer = self.Layers[i]
            
            OutValues = np.array([[nd.outvalue for nd in LastLayer.Nodes]])
            WeightMatrix = np.array([nd.weights for nd in LastLayer.Nodes])
            OutValues = OutValues.dot(WeightMatrix)
            
            for j in range(currLayer.nnodes):
                node = currLayer.Nodes[j]
                node.invalue = OutValues[0][j]
                    
                node.outvalue = self.ActivationFunction(node.invalue)
                
            LastLayer = currLayer
                
        return [node.outvalue for node in currLayer.Nodes]
    
    def BackProp(self,input,desiredOut):
        #Run one iteration of backwards propagation
        
        #Although typically described as a gradient descent, another interpretation of this
        #agorithm is as the pullback of a one-form of the difference between the desired and
        #actual output from the space of outputs to the space of weights.
        
        netout = self.RunNetwork(input) #run the error network
        derivo = np.array(desiredOut)-np.array(netout) #take difference between desired and actual output
        
        for i in range(self.Nlayers-2,-1,-1):
            currLayer = self.Layers[i]
            derivCurr = np.zeros(currLayer.nnodes,dtype=np.double) #to build derivative of output with respect to current layer
            
            dodx = np.array([self.ActivationDeriv(node.invalue) for node in self.Layers[i+1].Nodes], dtype=np.double)
            derivCurr = np.dot(np.array([nd.weights for nd in currLayer.Nodes]),dodx*derivo)
            
            for node in currLayer.Nodes:
                dEdw = derivo*dodx*node.outvalue
                dw = (1.0-self.alpha)*self.eta*dEdw + self.alpha*node.olddws
                node.weights += dw
                node.olddws = dw
                    
            derivo = derivCurr
                    
 

#Simple neural network with backpropagation learning
#Created by Mark Palenik
#I created this mainly as an exercise in learning python
#it's not necessarily the most efficient python code, as it is
#my first project, but it works.

import random
import math

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
                    for j in range(nNodes[i]): node.weights.append(2.0*(random.random()-1.0))
                    node.olddws = [0]*nNodes[i]
            
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
            for j in range(currLayer.nnodes):
                node = currLayer.Nodes[j]
                node.invalue = 0
                for k in range(LastLayer.nnodes):
                    node.invalue = node.invalue + LastLayer.Nodes[k].outvalue*LastLayer.Nodes[k].weights[j]
                    
                node.outvalue = self.ActivationFunction(node.invalue)
                
            LastLayer = currLayer
                
        return [node.outvalue for node in currLayer.Nodes]
    
    def BackProp(self,input,desiredOut):
        #Run one iteration of backwards propagation
        
        #Although typically described as a gradient descent, another interpretation of this
        #agorithm is as the pullback of a one-form of the difference between the desired and
        #actual output from the space of outputs to the space of weights.
        
        derivo = [] #derivative of error function with respect to the output of a given layer
        #start with the derivative of the error function with respect to the output layer
        netout = self.RunNetwork(input) #run the error network
        for i in range(len(input)): derivo.append(desiredOut[i] - netout[i]) #take difference between desired and actual output
        
        for i in range(self.Nlayers-2,-1,-1):
            currLayer = self.Layers[i]
            derivCurr = [0]*currLayer.nnodes #to build derivative of output with respect to current layer
            
            for k in range(self.Layers[i+1].nnodes):
                node = self.Layers[i+1].Nodes[k]
                dodx = self.ActivationDeriv(node.invalue) #derivative of node output with respect to input
                for j in range(currLayer.nnodes):
                    node2 = currLayer.Nodes[j]
                    #derivative of error with respect to node j
                    derivCurr[j] = derivCurr[j]+derivo[k]*dodx*node2.weights[k]
                    
                    dEdwj = derivo[k]*dodx*node2.outvalue #derivative of error with respect to weight
                    dw = (1.0-self.alpha)*self.eta*dEdwj + self.alpha*node2.olddws[k]
                    node2.weights[k] = node2.weights[k] + dw #adjust weight
                    node2.olddws[k] = dw #set momentum derivative
                    
            derivo = derivCurr
                    
 
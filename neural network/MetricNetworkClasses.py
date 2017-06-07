#Neural network with Metric-tensor learning
#Created by Mark Palenik
#This can be used as a more sophisticated replacement for NetworkClasses.py and
#NumpyNetworkClasses.  It utilizes numpy and adds a geometrically-based learning.
#A metric tensor gmn is defined by treating the weights as a complicated coordinate
#system in the space of output nodes.  The one-form defined by the difference between
#desired and actual output is pulled back to weight space, then converted to a vector.

#I created this to play around with another way of minimizing the error
#to see if it would improve over gradient descent.  Also, I wanted to practice using
#numpy.  Backwards propagation is included as well.

import numpy as np
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
            
    def MetricProp(self,input,desiredOut):
        #Define a metric tensor:
        #g_mn = dZ/du_m . dZ/du_n -> dO_i/dw_m dO_i/dw_n
        #and use it to change the one-forms in backwards propagation to vectors
        #and update based on the vector push-forward
        
        netout = self.RunNetwork(input) #run the error network
        derivo = np.array(desiredOut)-np.array(netout) #take difference between desired and actual output
        
        dimension=0
        for i in range(self.Nlayers-1):
            dimension = dimension+self.Layers[i].nnodes*self.Layers[i+1].nnodes
        
        nout = self.Layers[self.Nlayers-1].nnodes
        
        gmn = np.zeros((dimension,dimension))
        dOzdwkj = np.zeros((nout,dimension))
        dEdw = np.zeros(dimension)
        
        #dOzdOi = np.array([[w for w in node.weights] for node in self.Layers[self.Nlayers-2].Nodes])
        #dOzdOi=dOzdOi.T
        
        dOzdOi = np.identity(nout)
        
        offset=0
        offseto=0
        for l in range(self.Nlayers-2,-1,-1):
            currLayer = self.Layers[l]
            nextLayer = self.Layers[l+1]
            
            NewDeriv = np.zeros((nout,currLayer.nnodes))
            for i in range(nextLayer.nnodes):
                dOidx = self.ActivationDeriv(nextLayer.Nodes[i].invalue)
                for j in range(currLayer.nnodes):
                    dxdwji = currLayer.Nodes[j].outvalue     
                    dxdOj = currLayer.Nodes[j].weights[i]
                    
                    for z in range(nout):
                        NewDeriv[z][j] = NewDeriv[z][j] + dOzdOi[z][i]*dOidx*dxdOj
                        dOzdwkj[z][offset+j+i*currLayer.nnodes] = dOzdwkj[z][offset+j+i*currLayer.nnodes] + dOzdOi[z][i]*dOidx*dxdwji
                    
            for i in range(currLayer.nnodes*nextLayer.nnodes):
                for z in range(nout):
                    dEdw[i+offset] = dEdw[i+offset] + derivo[z]*dOzdwkj[z][i+offset]
                    
            dOzdOi = NewDeriv
            offset = offset + currLayer.nnodes * nextLayer.nnodes
            
        gmn = np.dot(dOzdwkj.T,dOzdwkj)
        #dEdw = np.dot(np.linalg.inv(gmn),dEdw)
        #dEdw = 0.1*dEdw/np.linalg.norm(dEdw)
        dEdw = np.dot(gmn,dEdw)
        
        indx=0
        for l in range(self.Nlayers-2,-1,-1):
            currLayer = self.Layers[l]
            nextLayer = self.Layers[l+1]
            for i in range(nextLayer.nnodes):
                for node in currLayer.Nodes:
                    dw = (1.0-self.alpha)*self.eta*dEdw[indx] + self.alpha*node.olddws[i]
                    node.weights[i] = node.weights[i]+dw
                    node.olddws[i] = dw
                    #node.weights[i] = node.weights[i]+10*dEdw[indx]
                    indx=indx+1
            
    def SquaredError(self,input,desiredout):
        netout = self.RunNetwork(input)
        MSE = 0.0
        for i in range(len(desiredout)): MSE = MSE + (netout[i]-desiredout[i])**2
        return MSE
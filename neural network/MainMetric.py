import NewtonNetworkClasses as nc
import pylab as pl #needed to plot the output

#simple demonstration of neural network
#use it to approximate y = 1/x and then retrain
#to approximate y = x

NeuralNet=nc.Network()
NeuralNet.alpha = 0.1 #adjust momentum.  Default is 0.1, but this is here to make it easier to adjust on the fly
NeuralNet.eta = 15.0  #adjust gradient scale factor default is 10.0

#This structure is not necessarily optimized for the problem
#I picked the number/size of the layers more or less randomly
NeuralNet.AddLayers([1,8,10,1])

#Some example outputs
print('Before training')
print (0.1,NeuralNet.RunNetwork([1]))
print (0.4,NeuralNet.RunNetwork([4]))
print (0.6,NeuralNet.RunNetwork([6]))

for i in range(3000):
    #NeuralNet.BackProp([1],[1])
    #NeuralNet.BackProp([2],[0.5])
    #NeuralNet.BackProp([8],[1.0/8.0])
    #NeuralNet.BackProp([4.7],[1.0/4.7])
    #NeuralNet.BackProp([3],[1.0/3.0])
    #NeuralNet.BackProp([7],[1.0/7.0])
    #NeuralNet.BackProp([6.4],[1.0/6.4])
    #NeuralNet.BackProp([1.3],[1.0/1.3])
    NeuralNet.MetricProp([1],[1])
    NeuralNet.MetricProp([2],[0.5])
    NeuralNet.MetricProp([8],[1.0/8.0])
    NeuralNet.MetricProp([4.7],[1.0/4.7])
    NeuralNet.MetricProp([3],[1.0/3.0])
    NeuralNet.MetricProp([7],[1.0/7.0])
    NeuralNet.MetricProp([6.4],[1.0/6.4])
    NeuralNet.MetricProp([1.3],[1.0/1.3])

#The same example outputs
print('After training')
print (1,NeuralNet.RunNetwork([1]))
print (4,NeuralNet.RunNetwork([4]))
print (6,NeuralNet.RunNetwork([6]))

#Make a plot of y=x and the neural network output vs. x
xlist = [(1.0+0.1*x) for x in range(0,90)]
ylist = []
for x in xlist: ylist.append(NeuralNet.RunNetwork([x]))
MSE = 0
for x in xlist: MSE = MSE + NeuralNet.SquaredError([x], [1.0/x])
print ('Mean Squared Error',MSE)

pl.plot(xlist,ylist)
pl.plot(xlist,[1.0/x for x in xlist])
pl.title('Network trained to produce y = 1/x.  Close to continue')
pl.show()

#use it to approximate a line y=x

#Clear network so weights are reset to random values
NeuralNet=nc.Network()
NeuralNet.alpha = 0.1 #adjust momentum.  Default is 0.1, but this is here to make it easier to adjust on the fly
NeuralNet.eta = 15.0  #adjust gradient scale factor default is 10.0

#Use the same structure as before, for no particular reason.
#It shows that the network is at least somewhat robust
NeuralNet.AddLayers([1,8,10,1])

#Some example outputs
print('Before training')
print (0.1,NeuralNet.RunNetwork([0.1]))
print (0.4,NeuralNet.RunNetwork([0.4]))
print (0.6,NeuralNet.RunNetwork([0.6]))

for i in range(3000):
    NeuralNet.MetricProp([0],[0])
    NeuralNet.MetricProp([0.1],[0.1])
    NeuralNet.MetricProp([1],[1])
    NeuralNet.MetricProp([0.2],[0.2])
    NeuralNet.MetricProp([0.7],[0.7])
    NeuralNet.MetricProp([0.3],[0.3])
    NeuralNet.MetricProp([0.9],[0.9])
    NeuralNet.MetricProp([0.6],[0.6])
    NeuralNet.MetricProp([0.8],[0.8])

#The same example outputs
print('After training')
print (0.1,NeuralNet.RunNetwork([0.1]))
print (0.4,NeuralNet.RunNetwork([0.4]))
print (0.6,NeuralNet.RunNetwork([0.6]))

#Make a plot of y=x and the neural network output vs. x
xlist = [0.1*x for x in range(0,10)]
ylist = []
for x in xlist: ylist.append(NeuralNet.RunNetwork([x]))

pl.plot(xlist,ylist)
pl.plot(xlist,xlist)
pl.title('Network trained to produce y = x.')
pl.show()




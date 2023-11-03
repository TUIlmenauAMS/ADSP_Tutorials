# -*- coding: utf-8 -*-
__author__ = 'Gerald Schuller'
__copyright__ = 'G.S.'

"""
Simple program to use a convolutional neural network to obtain a signal detector like matched filter,  using pytorch.
Gerald Schuller, November 2018.
"""

import torch 
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import sys

if sys.version_info[0] < 3:
   # for Python 2
   import cPickle as pickle
else:
   # for Python 3
   import pickle

# Device configuration
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def signal2pytorch(x):
   #Function to convert a signal vector s, like a mono audio signal, into a 3-d Tensor that conv1d of Pytorch expects,
   #https://pytorch.org/docs/stable/nn.html
   #conv1d Input: (N,Cin,Lin), Cin: numer of input channels (e.g. for stereo), Lin: length of signal, N: number of Batches (signals) 
   #Argument x: a 1-d signal as numpy array
   #output: 3-d Tensor X for conv1d input.
   X = np.expand_dims(x, axis=0)  #add channels dimension (here only 1 channel)
   X = np.expand_dims(X, axis=0)  #add batch dimension (here only 1 batch)
   X=torch.from_numpy(X)
   X=X.type(torch.Tensor)
   return X
    
class ConvNet(nn.Module):
   def __init__(self):
      super(ConvNet, self).__init__()
      self.detector=nn.Sequential(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=11, stride=1, padding=10, bias=False))
      #self.detector=nn.Conv1d(1, 1, 8, padding=8, stride=1, bias=False)
   
   def forward(self, x):
      out = self.detector(x)
      return out

if __name__ == '__main__':
    #Input signal X, a Ramp function:
    x= np.hstack((np.zeros(4),np.arange(0,1.1,0.1),np.zeros(5)))
    print("x=", x)
    X=signal2pytorch(x)
    y = np.zeros(30)
    y[16]=1 #Detecting the signal at its end (for convolution padding='causal'), like a matched filter.
    print("y=", y)
    Y=signal2pytorch(y)
    
    print("Input X[0,0,:]=", X[0,0,:], "X.shape=", X.shape )
    print("Target Y[0,0,:]=", Y[0,0,:], "Y.shape=", Y.shape)
    
    print("Generate Model:")
    #model = generate_model()     # Compile an neural net
    model = ConvNet()#.to('cpu')
    print("Def. loss function:")
    loss_fn = nn.MSELoss(size_average=False)
    #learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters())#, lr=learning_rate)
    
    for epoch in range(5000):
       Ydet=model(X)
       loss=loss_fn(Ydet, Y)
       if epoch%100==0:
          print(epoch, loss.item())
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
    
    torch.save(model.state_dict(), 'model_detector.torch')
    Ydet=model(X)
    Ydet=Ydet.data.numpy()
    #xnoisy=x+np.random.normal(size=x.shape)*0.1
    xnoisy=x+np.random.rand(20)-0.5
    Xnoisy=signal2pytorch(xnoisy)
    Ydetnoisy=model(Xnoisy)
    Ydetnoisy=Ydetnoisy.data.numpy()
    print("Predictions= ", Ydet[0,0,:])
    weights=list(model.parameters())
    print("Pytorch weights=", weights[0].data.numpy())
    
    print("Y=", Y)
    plt.plot(Ydet[0,0,:])
    plt.plot(Ydetnoisy[0,0,:])
    plt.legend(('For clean input', 'For noisy input'))
    plt.title('The Conv. Neural Network Output')
    plt.figure()
    plt.plot(weights[0].data.numpy()[0,0,:])
    plt.title('The Pytorch Weights, corr. instead of conv is used!')
    plt.figure()
    plt.plot(x)
    plt.plot(xnoisy)
    plt.legend(('Clean signal', 'Noisy signal'))
    plt.title('The Input Signal')
    plt.show()

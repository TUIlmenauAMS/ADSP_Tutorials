#%% LMS
import numpy as np
from sound import *
import matplotlib.pyplot as plt

x, fs = wavread('fspeech.wav')
#normalized float, -1<x<1
x = np.array(x,dtype=float)/2**15
print(np.size(x))
e = np.zeros(np.size(x))
xrek=np.zeros(np.size(x));
P=0;
L=10
h = np.zeros(L)
#have same 0 starting values as in decoder:
x[0:L]=0.0
quantstep=0.01;
for n in range(L, len(x)):
    if n> 4000 and n< 4010:
      print("encoder h: ", h, "e=", e)
    #prediction error and filter, using the vector of the time reversed IR:
    #predicted value from past reconstructed values:
    P=np.dot(np.flipud(xrek[n-L+np.arange(L)]), h)
    #quantize and de-quantize e to step-size 0.05 (mid tread):
    e[n]=np.round((x[n]-P)/quantstep)*quantstep;
    #Decoder in encoder:
    #new reconstructed value:
    xrek[n]=e[n]+P;
    #LMS update rule:
    h = h + 1.0* e[n]*np.flipud(xrek[n-L+np.arange(L)])

print("Mean squared prediction error:", np.dot(e, e) /np.max(np.size(e)))
#without quant.: 0.000215852452838
#with quant. with 0.01 : 0.000244936708861
#0.00046094397241
#quant with 0.0005: 0.000215872774695
print("Compare that with the mean squared signal power:", np.dot(x.transpose(),x)/np.max(np.size(x)))
print("The Signal to Error ratio is:", np.dot(x.transpose(),x)/np.dot(e.transpose(),e))
#The Signal to Error ratio is: 28.479576824, a little less than without quant.
#listen to it:
sound(2**15*e, fs)

plt.figure()
plt.plot(x)
#plt.hold(True)
plt.plot(e,'r')
plt.xlabel('Sample')
plt.ylabel('Normalized Sample')
plt.title('Least Mean Squares (LMS) Online Adaptation')
plt.legend(('Original','Prediction Error'))
plt.show()

# Decoder
h = np.zeros(L);
xrek = np.zeros(np.size(x));
for n in range(L, len(x)):
    if n> 4000 and n< 4010:
       print("decoder h: ", h)
    P=np.dot(np.flipud(xrek[n-L+np.arange(L)]), h)
    xrek[n] = e[n] + P 
    #LMS update:
    h = h + 1.0 * e[n]*np.flipud(xrek[n-L+np.arange(L)]);

plt.plot(xrek)
plt.xlabel('Sample')
plt.ylabel('Normalized Sample')
plt.title('The Reconstructed Signal')
plt.show()

#Listen to the reconstructed signal:
sound(2**15*xrek,fs)



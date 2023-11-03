#%% LMS
import numpy as np
from sound import *
import matplotlib.pyplot as plt

x, fs = wavread('fspeech.wav')
#normalized float, -1<x<1
x = np.array(x,dtype=float)/2**15
print(np.size(x))
e = np.zeros(np.size(x))

h = np.zeros(10)
#have same 0 starting values as in decoder:
x[0:10]=0.0
for n in range(10, len(x)):
    if n> 4000 and n< 4010:
      print("encoder h: ", h)
    #prediction error and filter, using the vector of the time reversed IR:
    e[n] = x[n] - np.dot(np.flipud(x[n-10+np.arange(0,10)]), h)
    #LMS update rule, according to the definition above:
    h = h + 1.0* e[n]*np.flipud(x[n-10+np.arange(0,10)])

print("Mean squared prediction error:", np.dot(e, e) /np.max(np.size(e)))
#0.000215852452838
print("Compare that with the mean squared signal power:", np.dot(x.transpose(),x)/np.max(np.size(x)))
#0.00697569381701
print("The Signal to Error ratio is:", np.dot(x.transpose(),x)/np.dot(e.transpose(),e))
#32.316954129056604, half as much as for LPC.

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
h = np.zeros(10);
xrek = np.zeros(np.size(x));
for n in range(10, len(x)):
    if n> 4000 and n< 4010:
       print("decoder h: ", h)
    xrek[n] = e[n] + np.dot(np.flipud(xrek[n-10+np.arange(10)]), h)
    #LMS update:
    h = h + 1.0 * e[n]*np.flipud(xrek[n-10+np.arange(10)]);


plt.plot(xrek)
plt.xlabel('Sample')
plt.ylabel('Normalized Sample')
plt.title('Reconstructed Signal')
plt.show()
#Listen to the reconstructed signal:
sound(2**15*xrek,fs)



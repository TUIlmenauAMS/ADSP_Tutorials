import numpy as np
def allp_delayfilt(tau):
    """
    Fractional-delay allpass filter design
    tau : desired delay in samples (float)
    returns: denominator a, numerator b
    """
    
    L = int(tau) + 1
    n = np.arange(L)

    a = np.cumprod((L-n)*(L-n-tau) / ((n+1)*(n+1+tau)))
    a = np.concatenate(([1.0], a))
    b = np.flipud(a)

    return a, b

if __name__ == '__main__':
   #testing the fractional delay allpass filter
   import matplotlib.pyplot as plt
   import scipy.signal as sp
   from freqresp import *
   #fractional delay of 5.5 samples:
   a,b=allp_delayfilt(5.5)
   print("a=",a,"b=",b)
   x=np.hstack((np.arange(4),np.zeros(8)))
   y=sp.lfilter(b,a,x) #applying the allpass filter
   plt.plot(x)
   plt.plot(y)
   plt.xlabel('Sample')
   plt.title('The IIR Fractional Delay Filter Result')
   plt.legend(('Original Signal', 'Delayed Signal'))
   plt.show()
   #Plot frequency response:
   freqresp(b,a) #omega=0.5: angle(H(0.5))=-2.8 Rad



import numpy as np


def allp_delayfilt(tau):
    '''
    produces a Fractional-delay All-pass Filter
    Arguments:tau = fractional delay in samples. When 'tau' is a float - sinc fu
nction. When 'tau' is an integer - just impulse.
    type of tau: float or int
    :return:
        a: Denumerator of the transfer function
        b: Numerator of the transfer function
    '''
    #L = max(1,int(tau)+1) with the +1 the max doesn't make sense anymore
    L = int(tau)+1
    n = np.arange(0,L)
    # print("n", n)

    a_0 = np.array([1.0])
    a = np.array(np.cumprod( np.divide(np.multiply((L - n), (L - n - tau)) , (np
.multiply((n + 1), (n + 1 + tau))) ) ))
    a = np.append(a_0, a)   # Denumerator of the transfer function
    # print("Denumerator of the transfer function a:", a)

    b = np.flipud(a)     # Numerator of the transfer function
    # print("Numerator of the transfer function b:", b)

    return a, b

if __name__ == '__main__':
   #testing the fractional delay allpass filter
   import matplotlib.pyplot as plt
   import scipy.signal as sp
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
   import freqz
   freqz.freqz(b,a) #omega=0.5: angle(H(0.5))=-2.8


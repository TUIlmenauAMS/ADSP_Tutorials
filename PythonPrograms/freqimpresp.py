import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, lfilter
    
def freqimpresp(b,a):
    #Impulse Response:
    try: #Test for IIR filter:
        isiir=(True if len(a) > 1 else False)
    except TypeError:
        isiir=False
    if isiir==False:
        plt.stem(b)
        plt.title('Impulse Response')
    else:
        x=np.zeros(80)
        x[0] = 1
        y=lfilter(b,a,x);
        plt.stem(y)
        plt.title('Impulse Response (Truncated)')
    plt.xlabel('Sample index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
    #Frequency Response:
    w,H=freqz(b,a)
    plt.subplot(2,1,1)
    plt.title('Frequency Response')
    plt.plot(w, 20 * np.log10(abs(H)), 'b')
    plt.ylabel('Amplitude (dB)')
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(w, np.angle(H), 'g')
    plt.ylabel('Angle (radians)')
    plt.xlabel('Normalized Frequency $\\Omega$ (rad)')
    plt.grid()
    plt.show()
    return()

if __name__ == '__main__':
    a=[1,-0.9]
    b=[1]
    freqimpresp(b,a)

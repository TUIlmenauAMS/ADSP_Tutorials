import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
    
def freqresp(b,a):
    w,H=freqz(b,a)
    plt.subplot(2,1,1)
    plt.title('Frequency Response')
    plt.plot(w, 20 * np.log10(abs(H)), 'b')
    plt.ylabel('Amplitude (dB)')
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(w, np.angle(H), 'g')
    plt.ylabel('Angle (radians)')
    plt.xlabel('Normalized Frequency')
    plt.grid()
    plt.show()
    return()

if __name__ == '__main__':
    a=[1,-0.9]
    b=[1]
    freqresp(b,a)

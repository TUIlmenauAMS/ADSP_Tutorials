# Module for show impulse rsponse answer
# Julia Peter, Mathias Kuntze
#Modified, Gerald Schuller, Nov. 2016

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp

def freqz(b, a=1, whole = False, axisFreqz = None, axisPhase = None):
    
    w, h = sp.freqz(b, a, worN=512, whole=whole)
    #w = w/np.pi
    fig = plt.figure()
    plt.title('Digital filter frequency response')
    plt.subplot(2,1,1)
    
    plt.plot(w, 20 * np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude (dB)')
    plt.xlabel('Normalized Frequency')
    plt.grid()
    if axisFreqz is not None:
        plt.axis(axisFreqz)
    
    plt.subplot(2,1,2)
    #angles = np.unwrap(np.angle(h))
    angles = np.angle(h)
    plt.plot(w, angles, 'g')
    plt.ylabel('Angle (radians)')
    plt.xlabel('Normalized Frequency')
    plt.grid()

    if axisPhase is not None:
        plt.axis(axisPhase)
    
    plt.show()
    return h

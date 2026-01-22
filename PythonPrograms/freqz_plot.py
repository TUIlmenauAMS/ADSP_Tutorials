# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 22:24:46 2017

@author: Robert
Modified: Gerald Schuller, Apr. 2017
"""

import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt

def freqz_plot(x, behaviour='matlab', color='b'):
    """ Plots the frequency response like freqz would do in Matlab/Octave would do
    Args:
        x           (np.array)          input signal
        behaviour   (str)               defines wether to behave like 'matlab' 
                                        (default) or 'octave' freqz plots
        color       (char)              default 'b' -> blue 
    """
    
    # get frequency bins and according magnitude values 
    f, h = sig.freqz(x)
 
    # normalized frequency
    fNorm = f/np.pi 
    
    # magnitude in dB
    hdB = 20 * np.log10(abs(h)+1e-5) #"+1e-5" avoids log10(0)!
    
    # open figure
    plt.figure()
    
    # octave also plots an extra zoomed version for the pass band
    if behaviour == 'octave':
        # Passband
        plt.subplot(311)
        plt.title('Passband')
        plt.plot(f, hdB, color)
        plt.axis([0,3.14,np.max(hdB)-6,np.max(hdB)+1])
        plt.grid(True)
        plt.xlabel('Normalized Frequency (rad/sample)')
        plt.ylabel('Magnitude (dB)')
    
    
    # Magnitude/Stopband
    if behaviour == 'octave':
        plt.subplot(312)
        plt.title('Stopband')
    else:
        plt.subplot(211)
        #plt.title('Magnitude')
    plt.plot(f, hdB, color)
    plt.axis([0,3.14,np.min(hdB)-1,np.max(hdB)+1])
    plt.grid(True)
    #plt.xlabel('Normalized Frequency (rad/sample)')
    plt.ylabel('Magnitude (dB)')
    
    # Phase
    if behaviour == 'octave':
        plt.subplot(313)
    else:
        plt.subplot(212)
    #plt.title('Phase')
    angles = np.angle(h)
    #angles = np.unwrap(np.angle(h)) #unwrapped version
    anglesGrad = (360 * angles)/(2*np.pi)
    plt.plot(f, anglesGrad, color)
    plt.axis([0,3.14,np.min(anglesGrad),np.max(anglesGrad)])
    plt.grid(True)
    plt.xlabel('Normalized Frequency (rad/sample)')
    plt.ylabel('Phase (degrees)')
    plt.show()

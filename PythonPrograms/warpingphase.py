# warpingphase from lecture 9
# Julia Peter, Mathias Kuntze
import numpy as np
def warpingphase(w, a):
    #produces phase wy for an allpass filter 
    #w: input vector of normlized frequencies (0..pi)
    #a: allpass coefficient
    #phase of allpass zero/pole :
    theta = np.angle(a); 
    #magnitude of allpass zero/pole :
    r = np.abs(a); 
    wy = -w-2*np.arctan((r*np.sin(w-theta))/(1-r*np.cos(w-theta)));    
    
    return wy
    
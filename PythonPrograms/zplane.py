#Gerald Schuller, June 2016
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def zplane(zeros, pole, axis = None):
    """Usage: zplane(zeros, poles)
    arguments:
    zeros: array of locations of zeros, complex valued
    poles: array of location of poles, complex valued
    returns: none
    plots the location of zeros and poles in the complex z-plane, with a unit circle.
    zeros are circles, poles are crosses.
    zeros, poles: array like, complex.
    """
    plt.figure()
    #Plotte die Pole in der komplexen Ebene als 'x':
    plt.plot(np.real(pole),np.imag(pole),'x')
    #Plotte die zeros als 'o':
    plt.plot(np.real(zeros),np.imag(zeros),'o')
    
    
    #passende Axen-Skalierung:
    plt.axis('equal')
    if axis is not None:
        plt.axis(axis)
    
    #Plot unit circle:
    circlere=np.zeros(512)
    circleim=np.zeros(512)
    for k in range(512):
       circlere[k]=np.cos(2*np.pi/512*k)
       circleim[k]=np.sin(2*np.pi/512*k)
    
    plt.plot(circlere,circleim)
    plt.legend(('Poles','Zeros'))
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    plt.title('Complex z-Plane')
    plt.show()
    return()

if __name__ == '__main__':
   #Testing, example:
   Fs=8000 #sampling freq.
   omega=2*np.pi*440.0 /Fs; #angle to real axis, 
   #normalized oscillation frequency of resulting time domain signal
   r=0.995 #radius, for a pole determines speed of decay of the 
   #resulting time domain signal, the closer to 1 the longer
   zeros=np.array([0.0])
   poles=np.array([r*np.exp(1j*omega), r*np.exp(-1j*omega)])
   zplane(zeros,poles)

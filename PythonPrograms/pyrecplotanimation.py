"""
Using Pyaudio, record sound from the audio device and plot, for 8 seconds, and display it live in a Window.
Usage example: python pyrecplotanimation.py
Gerald Schuller, October 2014 
"""

import pyaudio
import struct
#import math
#import array
import numpy as np
#import sys
#import wave
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import pylab
#import cv2
import sound


WIDTH = 2 #2 bytes per sample
CHANNELS = 1 #2
RATE = 44100  #Sampling Rate in Hz
CHUNK = int(2*RATE/50) #Blocksize, 2x period of 50hz
RECORD_SECONDS = 70

#Initialization of the plot window:
fig, ax = plt.subplots()
#plt.xlabel=("Seconds")
ax.set_xlabel('Time in Seconds')
ax.set_ylabel('Sample Value (Amplitude)')
ax.set_title('Short Time Block of Audio Signal')
x = np.arange(0, CHUNK,dtype=float)        # x-array
#Scale axis as this sine function:
line, = ax.plot(x/RATE, 10000.0*np.sin(x),'b.')
#line, = ax.plot(x/RATE, 10000.0*np.sin(x),'bo')
#line, = ax.plot(x/RATE, 10000.0*np.sin(x),'bs')
#line, = ax.plot(x/RATE, 10000.0*np.sin(x))

#Animation function:

def animate(i):
    # update the data
    #Reading from audio input stream into data with block length "CHUNK":
    data = stream.read(CHUNK)
    #Convert from stream of bytes to a list of short integers (2 bytes here) in "samples":
    #shorts = (struct.unpack( "128h", data ))
    shorts = (struct.unpack( 'h' * CHUNK, data ));
    samples=np.array(list(shorts),dtype=float);

    #plt.plot(samples)  #<-- here goes the signal processing.
    #line.set_ydata(np.log((np.abs(pylab.fft(samples))+0.1))/np.log(10.0))
    
    line.set_ydata(samples)
    return line,

#Initialization function:
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

#Main:
#Open audio device:
p = pyaudio.PyAudio()

a = p.get_device_count()
print("device count=",a)

for i in range(0, a):
    print("i = ",i)
    b = p.get_device_info_by_index(i)['maxInputChannels']
    print(b)
    b = p.get_device_info_by_index(i)['defaultSampleRate']
    print(b)

stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                #input_device_index=11,
                frames_per_buffer=CHUNK)


print("* recording")

#Do the animated plot by calling the animation function:
ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
    interval=25, blit=True)

plt.show()


# When everything done, release the capture

print("* done")

f.close()
stream.stop_stream()
stream.close()



"""
PyAudio Example: Make a modulation between input and output (i.e., record a
few samples, modulate them with a sine, and play them back immediately).
"""

import pyaudio
import struct
import math
#import array
import numpy as np
#import scipy
import matplotlib.pyplot as plt

CHUNK = 10240 #Blocksize
WIDTH = 2 #2 bytes per sample
CHANNELS = 1 #2
RATE = 32000  #Sampling Rate in Hz
RECORD_SECONDS = 8

def loglimit(x):
   #y=np.sign(x)*np.log(np.abs(x)+1)/np.log(32768)*32767;
   y=np.sign(x)*np.log(np.abs(1+255.0*x/32767.0)+1)/np.log(256.0)*32767.0;
   #x=x-0.5*x*x;
   #x=16*x;
   #if abs(x)< 10000:
   #   y=x;
   #else: 
   #   y=scipy.sign(x)*10000;
   
   return y;


f=plt.figure();
werte=np.arange(-32767,32676);
plt.plot(loglimit(werte));
f.show()

p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                #input_device_index=10,
                frames_per_buffer=CHUNK)
                

print("* recording")

#Loop for the blocks:
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    #Reading from audio input stream into data with block length "CHUNK":
    data = stream.read(CHUNK)
    #Convert from stream of bytes to a list of short integers (2 bytes here) in "samples":
    #shorts = (struct.unpack( "128h", data ))
    shorts = (struct.unpack( 'h' * CHUNK, data ));
    #samples=list(shorts);
    samples=np.array(list(shorts),dtype=float);
    #Compression function:
    sample=loglimit(samples)

    #converting from short integers to a stream of bytes in data:
    data=struct.pack('h' * len(samples), *samples);
    #Writing data back to audio output stream: 
    stream.write(data, CHUNK)

print("* done")

stream.stop_stream()
stream.close()

p.terminate()


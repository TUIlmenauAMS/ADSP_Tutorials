"""
PyAudio Example: Make a quantization between input and output (i.e., record a
few samples, quatize them with a mid-tread or mid-rise quantizer, and play them back immediately).
Using block-wise processing instead of a for loop
Gerald Schuller, Octtober 2014
"""

import pyaudio
import struct
#import math
#import array
import numpy as np
#import scipy

CHUNK = 5000 #Blocksize
WIDTH = 2 #2 bytes per sample
CHANNELS = 1 #2
RATE = 32000  #Sampling Rate in Hz
RECORD_SECONDS = 8

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
    samples=np.array(list(shorts),dtype=float);

    #start block-wise signal processing:



    ###mu-Law compression:###
    y=np.sign(samples)*(np.log(1+255*np.abs(samples/32768.0)))/np.log(256); #32768=2**15, 15 bits plus sign from the sound card

    ####Quantization, ####
    #16 steps for normalized range -1<=x<=1 (4 bits), normalized range for the formula
    q=2.0/16.0;  
    #Mid Tread quantization:
    indices=np.round(y/q)
    #Mid -Rise quantizer:
    #indices=np.floor(y/q)

    #### De-Quantization: #####
    #Mit-Tread:
    yrek=indices*q;
    #Mid -Rise quantizer:
    #yrek=indices*q+q/2;
    
    #no quantization:
    #yrek=y

    #### mu-law expanding function: ###
    #we use: exp(log(256)*yrek)=256^yrek
    samples=np.sign(yrek)*(np.exp(np.log(256)*np.abs(yrek))-1)/255*32768.0
    #end signal processing
    samples=np.clip(samples,-32000,32000)
    #converting from short integers to a stream of bytes in "data":
    data=struct.pack('h' * len(samples), *samples);
    #Writing data back to audio output stream: 
    stream.write(data, CHUNK)

print("* done")

stream.stop_stream()
stream.close()

p.terminate()


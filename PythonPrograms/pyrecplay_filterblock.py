"""
PyAudio Example: Filter the audio signal between input and output (i.e., record a
few samples, filter them, and play them back immediately).
Gerald Schuller, November 2014
"""

import pyaudio
import struct
import math
#import array
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

CHUNK = 1024 #Blocksize
WIDTH = 2 #2 bytes per sample
CHANNELS = 1 #2
RATE = 8000  #Sampling Rate in Hz
RECORD_SECONDS = 8

N=64
bpass=scipy.signal.remez(N, [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]  , [0.0, 1.0, 0.0], weight=[100.0, 1.0, 100.0])

#fig = plt.figure()
[freq, response] = scipy.signal.freqz(bpass)
plt.plot(freq, 20*np.log10(np.abs(response)+1e-6))
plt.xlabel('Normalized Frequency (pi is Nyquist Frequency)')
plt.ylabel("Magnitude of Frequency Response in dB")
plt.title("Magnitude of Frequency Response for our Bandbass Filter") 
plt.show()

plt.plot(bpass) 
plt.title('Impulse Response of our Bandpass Filter')
plt.show()

p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                #input_device_index=10,
                frames_per_buffer=CHUNK)
                

print("* recording")
#initialize memory for filter:
z=np.zeros(N-1)

#Loop for the blocks:
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    #Reading from audio input stream into data with block length "CHUNK":
    data = stream.read(CHUNK)
    #Convert from stream of bytes to a list of short integers (2 bytes here) in "samples":
    #shorts = (struct.unpack( "128h", data ))
    shorts = (struct.unpack( 'h' * CHUNK, data ));
    #samples=list(shorts);
    samples=np.array(list(shorts),dtype=float);
    #filter function:
    [filtered,z]=scipy.signal.lfilter(bpass, [1], samples, zi=z)
    filtered=np.clip(filtered, -32000,32000).astype(int)
    #converting from short integers to a stream of bytes in data:
    #comment this out to bypass filter:
    data=struct.pack('h' * len(filtered), *filtered);
    #Writing data back to audio output stream: 
    stream.write(data, CHUNK)

print("* done")

stream.stop_stream()
stream.close()

p.terminate()


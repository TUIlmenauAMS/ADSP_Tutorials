#Module for sound playback functions for pylab
#Gerald Schuller, October 9, 2013

from __future__ import print_function
import pyaudio
import struct
from numpy import clip
import numpy as np


def sound(audio,  samplingRate):
    """funtion to play back an audio signal, in array "audio", can be mono or stereo. 
    If stereo, each column of "audio" is one channel.
    usage: sound.sound(snd,  Fs)
    audio: array containing the audio wave
    sampligRate: The sampling rate for playback"""
  
    import pyaudio
    if len(audio.shape)==2:
       channels=audio.shape[1]
       if channels==2:
         print("Stereo")
       else:
          print("Mono")
    else:
       channels=1
       print("Mono")
    p = pyaudio.PyAudio()
    # open audio stream

    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=samplingRate,
                    output=True)

    sound = audio.astype(np.int16).tostring()
    stream.write(sound)

    # close stream and terminate audio object
    stream.stop_stream()
    stream.close()
    p.terminate()
    return  


def record(time, Fs, CHANNELS):
   """Records sound from a microphone to a vector s, for "time" seconds and with sampling rate of Fs samples/sec, also for stereo. E.g. for Mono with time=5, mono, and Fs=32000: import sound; snd=sound.record(5,32000,1)
   for Stereo: snd=sound.record(5,32000,2) """
   
   import numpy as np;
   global opened;
   global stream;
   CHUNK = 1000 #Blocksize
   WIDTH = 2 #2 bytes per sample
   #CHANNELS = 1 #2
   RATE = Fs  #Sampling Rate in Hz
   RECORD_SECONDS = time;

   p = pyaudio.PyAudio()

   a = p.get_device_count()
   print("device count=",a)
   
   for i in range(0, a):
      print("i = ",i)
      b = p.get_device_info_by_index(i)['maxInputChannels']
      print("max Input Channels=", b)
      b = p.get_device_info_by_index(i)['defaultSampleRate']
      print("default Sample Rate=", b)

   stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=False,
                #input_device_index=0,
                frames_per_buffer=CHUNK)
          
   print("* recording")
   snd=[];
   #Loop for the blocks:
   for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
      #print("Reading from audio input stream into data with block length CHUNK:")
      print(i)
      data = stream.read(CHUNK)
      #Convert from stream of bytes to a list of short integers (2 bytes here) in "samples":
      shorts = (struct.unpack( 'h' * CHANNELS*CHUNK, data ));
      #Reshape if stereo:
      shorts=np.reshape(shorts,(-1,CHANNELS))
      #samples = stream.read(CHUNK).astype(np.int16)
      try:
         snd=np.append(snd,shorts, axis=0);
      except ValueError:
         snd=shorts
   return snd;


import scipy.io.wavfile as wav

def wavread(sndfile):
   """This function implements a wavread function, similar to Octave or Matlab, to read a wav sound file into a vector x and sampling rate info 'Fs' at its return. It supports multi-channel audio. Use it with: import sound; [s,Fs]=sound.wavread('sound.wav'); or snd,Fs=sound.wavread('sound.wav')"""
   
   #Reading from an audio file: 
   Fs, x= wav.read(sndfile)
   return x, Fs
   
   
def wavwrite(snd,Fs,sndfile):
   """This function implements the wawwrite function, similar to Octave or Matlab, to write a wav sound file from a vector snd with sampling rate Fs, with: 
import sound; 
sound.wavwrite(snd,Fs,'sound.wav');"""

   #Writing to an audio file: 
   wav.write(sndfile,Fs,np.int16(snd))
   



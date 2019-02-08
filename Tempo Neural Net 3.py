import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import scipy.io.wavfile as wav
import scipy.signal as sig
from DataGen import *

from keras.models import Sequential
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten
from keras.optimizers import Adam

np.random.seed(1)

length = 44100
timestep = 441
features = length // timestep

max_pool = 4

#model mostly by Liam Schoneveld, with some edits by Tim Aris
model = Sequential()
model.add(Convolution1D(100, 3, padding='same', 
                    input_shape=(timestep, features)))
if max_pool > 1:
    model.add(Reshape((44100, 1))) 
    model.add(MaxPooling1D(pool_size=max_pool))
model.add(Activation('relu'))
model.add(Flatten())
model.summary()
model.compile(loss='mse', optimizer=Adam())


didntImprove = 0
cycles = 0
bestLoss = sys.maxsize
while(didntImprove < 10):
    Xt, Yt, songs, targets = dataGen(1000, length, timestep)
    Xv, Yv, songs, targets = dataGen(100, length, timestep)

    history = model.fit(Xt, Yt,
             batch_size=96, 
             epochs = 1,
             verbose=2, 
             validation_data=(Xv, Yv))
    
    cycles = cycles + 1
    if(bestLoss > history.history['val_loss'][0]):
        didntImprove = 0
        bestLoss = history.history['val_loss'][0]
    else:
        didntImprove = didntImprove + 1
        
print(cycles)
def getWholeSongOutput(model, song):
    start = 0
    output = []
    while((start + 1) * 44100 < len(song)):
        mat = np.matrix(song[start:start+44100])
        mat = mat.reshape(441, 100)
        array = np.array([mat])
        pred = model.predict(array)[0]
        for i in range(len(pred)):
            output.append(pred[i])
        
        start = start + 1
    output = np.array(output)
    return output

#function by user unutbu from stackoverflow
def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    #The following line takes too long for large inputs
    #assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

def peakToPeak(autoCor):
    plt.plot(autoCor)#[0:15000])
    plt.show()
    threshold = float(input("Enter threshold:"))
    for i in range(100, len(autoCor)):
        if autoCor[i] > threshold:
            return i
    
def guessTempo(song, model):
    output = getWholeSongOutput(model, song.signal)
    autoCor = estimated_autocorrelation(output)
    secondPeak = peakToPeak(autoCor)
    spb = secondPeak * 4 / 11025
    bps = 1 / spb 
    bpm = bps * 60
    print(song.title, "is at", bpm, "bpm.")
    return bpm
"""
for i in range(len(songs)):
    output = getWholeSongOutput(model, songs[i].signal)
    plt.plot(output)
    plt.plot(targets[i])
    plt.show()
"""

def outputPrediction(i, n, outputs, songs):
    song = maxPool(songs[i].signal)
    output = outputs[i]
    print(songs[i].title,":")
    plt.plot(song[n*11025:(n+1)*11025])
    plt.plot(output[n*11025:(n+1)*11025]*100)
    
outputs = []
for i in range(len(songs)):
    output = getWholeSongOutput(model, songs[i].signal)
    outputs.append(output)
    outputPrediction(i, 0, outputs, songs)


for i in range(len(songs)):
    autoCor = estimated_autocorrelation(outputs[i])
    secondPeak = peakToPeak(autoCor)
    spb = secondPeak * 4 / 11025
    bps = 1 / spb 
    bpm = bps * 60
    print(songs[i].title, "is at", bpm, "bpm.")


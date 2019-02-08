import os
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig
import matplotlib.pyplot as plt

class song():
    def __init__(self, tempo, title, path, signal):
        self.tempo = tempo
        self.title = title
        self.path = path
        self.signal = signal
        #self.target = target

def numberLen(string, pos):
    length = 0
    while(string[pos+length] in ['1','2','3','4','5','6','7','8','9','0']):
        length = length + 1
    return length

def getMatrix(signal, timestep, length, start, stop):
    snip = signal[start:stop]
    mat = np.matrix(snip)
    mat = mat.reshape(timestep, length // timestep)
    return mat      

def getTargetVector(tempo, length): #max pool?
    y = np.zeros(length // 4)
    bps = tempo / 60
    unitsPerBeat = 11025 / bps
    i = 0
    while(unitsPerBeat * i < length // 4):
        y[round(unitsPerBeat*i)] = 1
        i = i+1
    return y    

def maxPool(signal):
    result = []
    for i in range(len(signal) - 4):
        if(i%4 == 0):
            result.append(max(signal[i], signal[i+1], signal[i+2], signal[i+3]))
            
    return result

#def test(signal, target)


def dataGen(n, length, timestep):
    
    songLocation = os.getcwd() + '\\Songs'
    songs = []
    targets = []
    for filename in os.listdir(songLocation):
        tempo = int(filename[0:numberLen(filename, 0)])
        path = os.path.join(songLocation, filename)
        (fs, signal) = wav.read(path)
        signal = signal[:,0]                #44kHz
        signal = sig.decimate(signal, 4)    #11kHz
        target = getTargetVector(tempo, len(signal))
        
        targets.append(target)
        
        """
        print(filename, len(target))
        plt.plot(signal[60000:71025])
        plt.plot(target[60000:71025]*30000)
        plt.show()
        """
        
        s = song(tempo, filename, path, signal)
        songs.append(s)
    
    x = []
    y = []
    for i in range(n):
        r = np.random.randint(len(songs))       
        start = np.random.randint(len(targets[r])-length)
        stop = start + length
        
        target = targets[r][start:stop]
        target = maxPool(target)
        if(len(target) < 11025):
            target = target + (11025-len(target))*[0]
        target = np.array(target)
        
        xMat = getMatrix(songs[r].signal, timestep, length, start, stop)
        #yMat = getMatrix(songs[r].target, timestep, length, start, stop)
        x.append(xMat)
        y.append(target)
    
    return np.array(x), np.array(y), songs, targets
dataGen(2, 44100, 441)
#FILTER TERBARU
import scipy.signal
from scipy import signal
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import deque
from threading import Lock, Thread
import myo


class EmgCollector(myo.DeviceListener):
    """
    Collects EMG data in a queue with *n* maximum number of elements.
    """

    def __init__(self, n):
        self.n = n
        self.lock = Lock()
        self.emg_data_queue = deque(maxlen=n)

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)

    # myo.DeviceListener

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        with self.lock:
            self.emg_data_queue.append((event.timestamp, event.emg))


#PROSES SEMUA DATA

## filtering
low = 0.5/100
high = 50/100
pole = 5
samp_freq = 200 
notch_freq =60.0  
quality_factor = 100.0 

#winddwing
n_steps  = 40
n_chanel = 8

def filters (chanel,pole,low,high):
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
    dn = signal.filtfilt(b_notch, a_notch, chanel)
    b, a = scipy.signal.butter(pole, [low, high], 'band')
    df = scipy.signal.lfilter(b, a, dn)
    dframe = pd.DataFrame(df)
    
    return dframe

# WINDOWING TERBARU

def split_sequences(sequences, n_steps):
	X = list()
	for i in range(len(sequences)):
		end_ix = i + n_steps
		if end_ix > len(sequences):
			break
		seq_x = sequences[i:end_ix]
		X.append(seq_x)
		
	return np.array(X)

# FITUR RMS

def rms (a):
    out = np.sqrt(np.mean(np.square(a), axis = 1))
  
    return out

#PREP
frame = 40
def prep(x)  :
    
    ch1  = filters (x.ch1,pole,low,high)
    ch2  = filters (x.ch2,pole,low,high)
    ch3  = filters (x.ch3,pole,low,high)
    ch4  = filters (x.ch4,pole,low,high)
    ch5  = filters (x.ch5,pole,low,high)
    ch6  = filters (x.ch6,pole,low,high)
    ch7  = filters (x.ch7,pole,low,high)
    ch8  = filters (x.ch8,pole,low,high)

    datafil = pd.concat([ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8],axis=1)
    datafil.columns= ['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']
    
    h1  = datafil['ch1'].values
    h2  = datafil['ch2'].values
    h3  = datafil['ch3'].values
    h4  = datafil['ch4'].values
    h5  = datafil['ch5'].values
    h6  = datafil['ch6'].values
    h7  = datafil['ch7'].values
    h8  = datafil['ch8'].values
    

    c1 = h1.reshape(len(h1),1)
    c2 = h2.reshape(len(h2),1)
    c3 = h3.reshape(len(h3),1)
    c4 = h4.reshape(len(h4),1)
    c5 = h5.reshape(len(h5),1)
    c6 = h6.reshape(len(h6),1)
    c7 = h7.reshape(len(h7),1)
    c8 = h8.reshape(len(h8),1)
    
    
    seg =  np.hstack((c1,c2,c3,c4,c5,c6,c7,c8))
    x = split_sequences(seg, n_steps)
    
    
    #xr = x.reshape(-1,n_chanel*n_steps)
      
    out= []
    for a in  x :
        df = pd.DataFrame(a)
        df.columns = ['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']
                
        mav1 = np.sum(np.absolute(df["ch1"].values)) / frame 
        mav2 = np.sum(np.absolute(df["ch1"].values)) / frame 
        mav3 = np.sum(np.absolute(df["ch1"].values)) / frame  
        mav4 = np.sum(np.absolute(df["ch1"].values)) / frame  
        mav5 = np.sum(np.absolute(df["ch1"].values)) / frame  
        mav6 = np.sum(np.absolute(df["ch1"].values)) / frame 
        mav7 = np.sum(np.absolute(df["ch1"].values)) / frame  
        mav8 = np.sum(np.absolute(df["ch1"].values)) / frame  
        mav_total = np.hstack((mav1,mav2,mav3,mav4,mav5,mav6,mav7,mav8))
    
        out.append(mav_total)

    data_mav       = np.array(out)

    
    
    
    return data_mav
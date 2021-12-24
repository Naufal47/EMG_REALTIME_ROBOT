import serial
import joblib
import pandas as pd
import numpy as np
import myo
import matplotlib.pyplot as plt
import pyautogui as pg
import time
import os
import sys
import struct

#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import Normalizer


from libMav import *
#MinMaxScaler(feature_range=(0, 1))
model1 = joblib.load('jempolNaufalMAV.pkl')
model2 = joblib.load('telunjukNaufalMAV.pkl')
model3 = joblib.load('tengahNaufalMAV.pkl')
model4 = joblib.load('manisNaufalMAV.pkl')
model5 = joblib.load('kelingkingNaufalMAV.pkl')
#modelscaler = joblib.load('scaler.pkl')
arduino = serial.Serial('com6',9600)
#modelpca = joblib.load('pcawindowbaru.pkl')
#arduino = serial.Serial("com10", 9600)
#print(modelscaler)
#print(modelscaler)
class Plot(object):
    def __init__(self, listener):
        self.n = listener.n
        self.listener = listener
        #self.fig = plt.figure()
        #self.axes = [self.fig.add_subplot('81' + str(i)) for i in range(1, 9)]
        #[(ax.set_ylim([-100, 100])) for ax in self.axes]
        #self.graphs = [ax.plot(np.arange(self.n), np.zeros(self.n))[0] for ax in self.axes]
        #plt.ion()
    

    ##def update_plot(self):
        #print('isisne kosong')
        #for g, data in zip(self.graphs, emg_data):
           
        #plt.tight_layout()
        #plt.draw()
    
    def main(self):
        datahasil = []
        for i in range(1000):

            #self.update_plot()
            plt.pause(1/4) 
            emg_data = self.listener.get_emg_data()
            emgX1 = ([x[1] for x in emg_data])
            emgX = pd.DataFrame(emgX1)

            oi = emgX.shape
            



            if oi[0] > 39:
                data = np.array(emgX).reshape(-1,8)

                #print(data)
                #emg = np.array(data).reshape(-1,8)

                save_emg = pd.DataFrame(data)
                save_emg.columns = ['ch1', 'ch2', 'ch3','ch4','ch5','ch6','ch7','ch8']
                #print(save_emg)
                datafitur = prep(save_emg)
                #print(datafituran)
                #X1 = modelscaler.transform(datafituran)
                
                              
                realtime1 = model1.predict(datafitur)
                realtime2 = model2.predict(datafitur)
                realtime3 = model3.predict(datafitur)
                realtime4 = model4.predict(datafitur)
                realtime5 = model5.predict(datafitur)
                #print(realtime1, realtime2, realtime3, realtime4, realtime5)
                a = np.array(realtime1).reshape(-1,1)
                b = np.array(realtime2).reshape(-1,1)
                c = np.array(realtime3).reshape(-1-1)
                d = np.array(realtime4).reshape(-1,1)
                e = np.array(realtime5).reshape(-1,1)

                f = np.min(a)
                g = np.min(b)
                h = np.min(c)
                k = np.min(d)
                l = np.min(f)
                ai=(np.abs((a-f))*2)[0]
                bi=(np.abs((b-g))*2)[0]
                ci=(np.abs((c-h))*2)[0]
                di=(np.abs((d-k))*2)[0]
                ei=(np.abs((e-l))*2)[0]

                k1 = int(np.round(ai,1))
                k2 = int(np.round(bi,1))
                k3 = int(np.round(ci,1))
                k4 = int(np.round(di,1))
                k5 = int(np.round(ei,1))

                sudahurutan = np.vstack((ai,bi,ci,di,ei)).reshape(1,5)
                print(k1,k2,k3,k4,k5)
                #print(arduino.read())
                #datakirim = arduino.write(wesurutan)
                #arduino.flush()
                time.sleep(1)
                arduino.write(struct.pack('>BBBBB',k1,k2,k3,k4,k5))
                
               
                    #if ii < 30:
                        #pg.keyDown('a')
                        #time.sleep(1)
                        #pg.keyDown('b')
                        #time.sleep(1)
                        #pg.keyDown('d')
                        #time.sleep(1)
                    #pg.keyDown('g')
                    #time.sleep(1)
                    #pg.keyUp('g')
                    #time.sleep(1)
                    #pg.keyDown('g')
                    #time.sleep(1)
                    #pg.keyUp('g')
                    #time.sleep(1)
                    #pg.keyUp('a')
                
                #aw = arduino.readline().encode('ascii')
                #aw.write(wesurutan)
                #oii = pd.DataFrame(wesurutan).head(1)
                #print(wesurutan) 
                #arduino.write(oii)
                #time.sleep(1)
                #oii = np.array(oii).tolist()
                #datahasil.append(oii)
                
                

            #datahasil1 = pd.DataFrame(np.array(datahasil).reshape(-1,5))
            #print(datahasil1.shape)
            #datahasil1.to_csv('HASILMAVterbaru5.csv')
           
        
                 
                
                
                
                #print(realtime1,realtime2,realtime3,realtime4,realtime5)
                #while True:
                    #os.startfile("tangan baru 2.blend")
                    #oi = datapred.mean()
                    #print(oi)
                    #if oi < 20:
                        #pg.keyDown('a')
                        #time.sleep(2)
                    #if oi > 40 :
                        #pg.keyDown('b')
                        #time.sleep(6)
                        #pg.keyDown('d')
                        #time.sleep(6)
                        #pg.keyDown('g')
                        #time.sleep(6)
                    
                    
               

                

def main():
    myo.init()
    hub = myo.Hub()
    listener = EmgCollector(80)
    with hub.run_in_background(listener.on_event):
        Plot(listener).main()
    
   
#while True :
    #main()

if __name__ == '__main__':
    main()
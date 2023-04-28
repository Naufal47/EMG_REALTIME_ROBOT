# https://github.com/NiklasRosenstein/myo-python
# The MIT License (MIT)
#
# Copyright (c) 2017 Niklas Rosenstein
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

from matplotlib import pyplot as plt
from collections import deque
from threading import Lock, Thread

import myo
import numpy as np
import pandas as pd
import pyautogui
import keyboard
from lib_prosesing_myo import *
import serial

arduino = serial.Serial('com6', 9600)


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

class Plot(object):
    def __init__(self, listener):
        self.n = listener.n
        self.listener = listener
        # self.fig = plt.figure()
        # self.axes = [self.fig.add_subplot('81' + str(i)) for i in range(1, 9)]
        # [(ax.set_ylim([-100, 100])) for ax in self.axes]
        # self.graphs = [ax.plot(np.arange(self.n), np.zeros(self.n))[0] for ax in self.axes]
        # plt.ion()
    
    def update_plot(self):
          # Mendapatkan posisi mouse awal
        prev_x, prev_y = pyautogui.position()

            # Setel durasi interval antara input keyboard (dalam detik)
        interval = 0.1    
        while True:
            emg_data = self.listener.get_emg_data()

            emg_data = np.array([x[1] for x in emg_data]).T

            save_emg = pd.DataFrame(emg_data)
            save_emgTranspose = save_emg.T
            save_emgTranspose = pd.DataFrame(save_emgTranspose)
            # print(type(save_emgTranspose))
            datanya = np.array(save_emgTranspose)
            datanya1 = np.abs(datanya)
            datanya2 = np.mean(datanya1)
            arduino.write(datanya2)
            print(datanya2)

     
            # Mendapatkan posisi mouse saat ini
            current_x, current_y = pyautogui.position()

            # Menghitung perubahan posisi mouse
            dx = current_x - prev_x
            dy = current_y - prev_y

            # Mengonversi perubahan posisi mouse menjadi input keyboard
            
            if dx > 0:
                keyboard.press('d')  # Mengganti right menjadi d
                keyboard.release('d')
            elif dx < 0:
                keyboard.press('a')  # Mengganti left menjadi a
                keyboard.release('a')

            if dy > 0:
                keyboard.press('s')  # Mengganti down menjadi s
                keyboard.release('s')
            elif dy < 0:
                keyboard.press('w')  # Mengganti up menjadi w
                keyboard.release('w')

            if datanya2 > 11:
                if counter == 0:
                    keyboard.press('m')
                    keyboard.release('m')
                    counter += 1
                elif counter > 0 and counter < 6:

                # Jika nilai rata-rata di atas 11 dan counter > 0 dan < 6, tambahkan counter
                    counter += 1
                else:
                # Jika nilai rata-rata di atas 11 dan counter >= 6, tekan tombol "m" lagi
                    keyboard.press('m')
                    keyboard.release('m')
                    counter = 0
            else:
                counter = 0  # Reset counter jika nilai rata-rata di bawah 11
                keyboard.press('n')  # Mengganti right menjadi d
                # keyboard.release('m')


            #elif datanya2 < 3:
                #keyboard.press('n')  # Mengganti right menjadi d
                #keyboard.release('n') 

            # Menyimpan posisi mouse saat ini sebagai posisi mouse sebelumnya
            prev_x, prev_y = current_x, current_y

            # Menunggu interval sebelum mengambil posisi mouse berikutnya
            pyautogui.sleep(interval)
        
        
       
        # for g, data in zip(self.graphs, emg_data):
        #     if len(data) < self.n:
        #         # Fill the left side with zeroes.
        #         data = np.concatenate([np.zeros(self.n - len(data)), data])
        #     g.set_ydata(data)
        # #plt.tight_layout()
        # plt.draw()
        

    def main(self):
        while True:
            self.update_plot()
            plt.pause(1.0 / 30)

def main():
    myo.init()
    hub = myo.Hub()
    listener = EmgCollector(40)

    with hub.run_in_background(listener.on_event):
        Plot(listener).main()


if __name__ == '__main__':
    main()
    #with hub.run_in_background(listener.on_event):
        # Plot(listener).main()
        #while True:
            #if len(listener.get_emg_data()) > 0:
                #break    
        
        # for i in range (10):
        #     # for i in range(10):
        #emg_data = listener.get_emg_data()    
        # data = listener.get_emg_data()
        #print(emg_data)
        #print('*'*100)

import numpy as np
import scipy.fft as fft

import matplotlib.pyplot as plt

##


fileName = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Wave-ParticleDuality/rawData/0-0-none-test1.txt'
data = np.loadtxt(fileName, unpack = True, skiprows=1)

dErr = 0
dErr = [np.ones(len(data[0]))]
for i in range(2, 4):
    errors = [np.sqrt(data[i][j]) for j in range(len(data[i]))]
    dErr.append(np.asarray(errors))

dErr = np.asarray(dErr)

##




yf = fft.rfft(data[2])
xf = fft.rfftfreq( len(data[0]), .1)


plt.clf()

plt.plot(xf[:50], np.abs(yf[:50]))

plt.show()

##
plt.clf()

plt.plot(data[0], data[2], 'b.')
plt.plot(data[0], data[2][0] * np.cos( 0.08 * 2 * np.pi *data[0] - .8) / 2.5 + data[2][0])

plt.show()
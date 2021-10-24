##preamble

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

## Data Induction
print(energies) #generated by Attenuation.py -- run those cells in the same kernel as this script

NistDataPath = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/GammaX-Sections/nist_mu_al.txt'

savePath = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/GammaX-Sections/AttenuationComparison.png'

NistVals = np.loadtxt(NistDataPath, unpack=True, skiprows=1)

print(NistVals[:, :3])
NistVals[1] *= .1 #unit conversion: cm^{-1} -> mm^{-1}
print(NistVals[:, :3])


#find bounds of relevant data
emin = min(energies)
emax = max(energies)

eminIndex = 0
emaxIndex = 0

for i, e in enumerate(NistVals[0]):
    if(e >= emin and eminIndex == 0):
        eminIndex = i - 4
    if(e > emax):
        emaxIndex = i +2
        break

print(eminIndex, emaxIndex)

#programming uncertainties in NIST data

refUncertainty = .03 #from wiki, assume 3% error on all NIST values

NISTUB = NistVals[1] * (1 + refUncertainty)
NISTLB = NistVals[1] * (1 - refUncertainty)




## Plotting Attenuation Coefficients
plt.clf()

plt.figure(figsize=(8,6))
plt.title('Measured Attenuation Coefficients vs NIST Data')
plt.xlabel('Energy (keV)')
plt.xscale('log')
plt.ylabel('Attenuation Coefficient (mm$^{-1}$)')
plt.yscale('log')

ThomsonY = []
for i in range(len(NistVals[0][eminIndex:emaxIndex])):
    ThomsonY.append(0.052)


pCent, = plt.plot(NistVals[0][eminIndex:emaxIndex], NistVals[1][eminIndex:emaxIndex], 'k-', lw='.5', markersize=1)
pUB, = plt.plot(NistVals[0][eminIndex:emaxIndex], NISTUB[eminIndex:emaxIndex], ls='--', color='gray', lw='.5', markersize=1)
pLB, = plt.plot(NistVals[0][eminIndex:emaxIndex], NISTLB[eminIndex:emaxIndex], ls='--', color='gray', lw='.5', markersize=1)
vals = plt.errorbar(energies, coeffs, dcoeffs, fmt='r.', lw=1, markersize=2, capsize=3)
Thomson, = plt.plot(NistVals[0][eminIndex:emaxIndex], ThomsonY, 'b--', lw=.5)

plt.legend( [pCent, (pUB, pLB), vals, Thomson], ['NIST Reported Values', 'NIST Values Error bounds', 'Measured Values', 'Thomson Scattering'], handler_map={tuple: HandlerTuple(ndivide=None)}, loc='upper right')
plt.savefig(savePath)
plt.show()


## Linear Interpolation and T calculation

def interpolate(energy):
    #find which two points 'energy' is between:
    j1 = 0
    j2 = 0
    for i in range(len(NistVals[0])):
        if NistVals[0][i] >= energy:
            j1 = i-1
            j2 = i
            break


    #interpolate lambda
    t = (float(energy) - NistVals[0][j1]) / (NistVals[0][j2] - NistVals[0][j1])
    out = t * NistVals[1][j2] + (1-t)* NistVals[1][j1]
    return out

def calcT(energy, emp, dEmp):
    ref = interpolate(energy)
    dRef = .03 * ref
    print(1000 * dEmp)
    print(1000 * dRef) #change units mm^{-1} -> m^{-1}

    out = (float(emp) - ref) / np.sqrt( dEmp**2 + dRef **2 )
    print(out)
    return out


lambdaE = np.asarray([365, 213, 48.8, 29.0, 19.6, 18.7, 18.5])
dLambdaE = np.asarray([55, 2.5, 1.1, 1.2, .58, 2.1, 1.65])


lambdaR = np.asarray([279, 279, 54.1, 26.6, 22.6, 20.2, 14.7])
dLambdaR = np.asarray([8.4, 8.4, 1.6, .8, .68, .61, .44])

scaleFactor = 1.2792579465941572 * .01 #unit conversion m^{-1} -> cm^{-1}

sigma = [0,0,0,0,0,0,0]
dSigma = [0,0,0,0,0,0,0]

for i in range(len(lambdaE)):
    print('line')
    print(lambdaE[i] * scaleFactor, dLambdaE[i] * scaleFactor)
    sigma[i] = lambdaE[i] * scaleFactor
    dSigma[i] = dLambdaE[i] * scaleFactor
    print(lambdaR[i] * scaleFactor, dLambdaR[i] * scaleFactor)
    print('')

thomsonX = .6652

for i in range(len(lambdaE)):
    print('line %d'%i)
    print('t = %f'%((sigma[i] - thomsonX) / dSigma[i]) )

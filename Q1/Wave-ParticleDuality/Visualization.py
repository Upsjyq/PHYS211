import numpy as np
import scipy.fft as fft
from scipy import optimize

import matplotlib.pyplot as plt


##function definition


def residual(p,func, xvar, yvar, err):
    return (func(p, xvar) - yvar)/err

# The code below defines our data fitting function.
# Inputs are:
# initial guess for parameters p0
# the function we're fitting to
# the x,y, and dy variables
# tmi can be set to 1 or 2 if more intermediate data is needed

def data_fit(p0,func,xvar, yvar, err,tmi=0):
    try:
        fit = optimize.least_squares(residual, p0, args=(func,xvar, yvar, err),verbose=tmi)
    except Exception as error:
        print("Something has gone wrong:",error)
        return p0, np.zeros_like(p0), -1, -1
    pf = fit['x']

    print()

    try:
        cov = np.linalg.inv(fit['jac'].T.dot(fit['jac']))
        # This computes a covariance matrix by finding the inverse of the Jacobian times its transpose
        # We need this to find the uncertainty in our fit parameters
    except:
        # If the fit failed, print the reason
        print('Fit did not converge')
        print('Result is likely a local minimum')
        print('Try changing initial values')
        print('Status code:', fit['status'])
        print(fit['message'])
        return pf,np.zeros_like(pf), -1, -1
            #You'll be able to plot with this, but it will not be a good fit.

    chisq = sum(residual(pf,func,xvar, yvar, err) **2)
    dof = len(xvar) - len(pf)
    red_chisq = chisq/dof
    pferr = np.sqrt(np.diagonal(cov)) # finds the uncertainty in fit parameters by squaring diagonal elements of the covariance matrix
    print('Converged with chi-squared {:.2f}'.format(chisq))
    print('Number of degrees of freedom, dof = {:.2f}'.format(dof))
    print('Reduced chi-squared {:.2f}'.format(red_chisq))
    print()
    Columns = ["Parameter #","Initial guess values:", "Best fit values:", "Uncertainties in the best fit values:"]
    print('{:<11}'.format(Columns[0]),'|','{:<24}'.format(Columns[1]),"|",'{:<24}'.format(Columns[2]),"|",'{:<24}'.format(Columns[3]))
    for num in range(len(pf)):
        print('{:<11}'.format(num),'|','{:<24.3e}'.format(p0[num]),'|','{:<24.3e}'.format(pf[num]),'|','{:<24.3e}'.format(pferr[num]))
    return pf, pferr, chisq, dof

# ## Declare/Empty Relevant Variables
# data = []
# dErr = []
# angles = [] #manual axis, motorized axis, external axis


##

dataSave = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Wave-ParticleDuality/rawData/dataCompiled.npy'

errSave = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Wave-ParticleDuality/rawData/errCompiled.npy'

angleSave = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Wave-ParticleDuality/rawData/anglesCompiled.npy'


# ## Data Induction
#
# trialNum = 0
#
# fileName = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Wave-ParticleDuality/rawData/20-65-none-test1.txt'
#
# dataRaw = np.loadtxt(fileName, unpack = True, skiprows=1)
#
# dErrRaw = [np.zeros(len(dataRaw[0]))]
# for i in range(2, 4):
#     errors = [np.sqrt(dataRaw[i][j]) for j in range(len(dataRaw[i]))]
#     dErrRaw.append(np.asarray(errors))
#
# data.append(dataRaw)
# dErr.append(dErrRaw)
# angles.append([20, 65, 'NONE'])

# ## Saving compiled data for re-induction
#
#
# np.save(dataSave, data)
#
# np.save(errSave, dErr)
#
# np.save(angleSave, angles)

## Data re-induction

data = np.load(dataSave)

err = np.load(errSave)

angles = np.load(angleSave)

guess = [[500, 0, 1500] for i in range(len(data))] #amp, phase, offset



## DFT


yf = fft.rfft(data[2])
xf = fft.rfftfreq( len(data[0]), .1)


plt.clf()

plt.plot(xf[:50], np.abs(yf[:50]))

plt.show()

## Fitting sinusoids to fringes

freqFit = .5163 #determined by just fitting sinusoids
def sinusoid(p, x):
    return p[0] * np.cos( freqFit * x + p[1] ) + p[2]


pf, pferr, chisq, dof = [[],[]], [[],[]], [[],[]], [[],[]]


# The first data line, coinc 1&2
for i, e in enumerate(data):
    pf0, pferr0, chisq0, dof0 = data_fit(guess[i], sinusoid, e[0], e[2], err[i][1])

    plt.clf()
    plt.errorbar(e[0], e[2], err[i][1])
    xSmooth = np.linspace(e[0][0], e[0][-1], num=100)
    plt.plot(xSmooth,   (sinusoid(pf0, xSmooth))   )

    print(chisq0)

    plt.show()

    text = input('continue? (y/n/adjustGuess)')
    if text == 'y':
        pass
    elif text == 'a':
        command = input('New guess[0]:')
        guess[i][0] = float(command)
        command = input('New guess[1]:')
        guess[i][1] = float(command)
        command = input('New guess[2]:')
        guess[i][2] = float(command)
    else:
        break

    pf[0].append(pf0)
    pferr[0].append(pferr0)
    chisq[0].append(chisq0)
    dof[0].append(dof0)


# second data line, coinc 1&3
for i, e in enumerate(data):
    pf0, pferr0, chisq0, dof0 = data_fit(guess[i], sinusoid, e[0], e[3], err[i][2])

    plt.clf()
    plt.errorbar(e[0], e[3], err[i][2])
    xSmooth = np.linspace(e[0][0], e[0][-1], num=100)
    plt.plot(xSmooth,   (sinusoid(pf0, xSmooth))   )

    print(chisq0)

    plt.show()

    text = input('continue? (y/n/adjustGuess)')
    if text == 'y':
        pass
    elif text == 'a':
        command = input('New guess[0]:')
        guess[i][0] = float(command)
        command = input('New guess[1]:')
        guess[i][1] = float(command)
        command = input('New guess[2]:')
        guess[i][2] = float(command)
    else:
        break

    pf[1].append(pf0)
    pferr[1].append(pferr0)
    chisq[1].append(chisq0)
    dof[1].append(dof0)


## save fit data

fitParamsSave = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Wave-ParticleDuality/rawData/fitParams.npy'
fitErrSave = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Wave-ParticleDuality/rawData/fitErr.npy'
fitChisqSave = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Wave-ParticleDuality/rawData/fitChisq.npy'
fitDOFSave = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Wave-ParticleDuality/rawData/fitDOF.npy'


##
np.save(fitParamsSave, pf)
np.save(fitErrSave, pferr)
np.save(fitChisqSave, chisq)
np.save(fitDOFSave, dof)

## re-induct fit data

pf = np.load(fitParamsSave)
pferr = np.load(fitErrSave)
chisq = np.load(fitChisqSave)
dof = np.load(fitDOFSave)


## Plotting, experimental



## Plotting (Output)
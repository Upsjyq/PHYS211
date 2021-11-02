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

## Declare/Empty Relevant Variables
data = []
dErr = []
angles = [] #manual axis, motorized axis, external axis


##

dataSave = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Wave-ParticleDuality/rawData/dataCompiled.npy'

errSave = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Wave-ParticleDuality/rawData/errCompiled.npy'

angleSave = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Wave-ParticleDuality/rawData/anglesCompiled.npy'


## Data Induction

trialNum = 0

for dataNum in range(1,19):
    fileName = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Wave-ParticleDuality/rawData/Day2-test2/0-45-%d0.txt'%dataNum

    dataRaw = np.loadtxt(fileName, unpack = True, skiprows=1)

    dErrRaw = [np.zeros(len(dataRaw[0]))]
    for i in range(2, 4):
        errors = [np.sqrt(dataRaw[i][j]) for j in range(len(dataRaw[i]))]
        dErrRaw.append(np.asarray(errors))

    data.append(dataRaw)
    dErr.append(dErrRaw)
    angles.append([0,45,10 * dataNum])

## Saving compiled data for re-induction


np.save(dataSave, data)

np.save(errSave, dErr)

np.save(angleSave, angles)

## Data re-induction

data = np.load(dataSave)

err = np.load(errSave)

angles = np.load(angleSave)

guess = [[500, 0, 1500] for i in range(len(data))] #amp, phase, offset


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


##

for i in range(len(pf[0])):
    print(len(pf[0][i]))
pf = np.asarray(pf)
pferr = np.asarray(pferr)
chisq = np.asarray(chisq)
dof = np.asarray(dof)


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

'''
pf[0][3][2] is the 3rd fit parameter (offset) of the 4th trial (0-45-55-test1), to the zero-th data channel (coincidence rate 1&2).

similar for the others.
'''


## process/ plot no-external-filter runs
'''
Splitting data into two categories: the runs without external polarizing filter, and the runs with external polarizing filter
'''
externBool = []
for i, e in enumerate(angles):
    if e[2] == 'NONE':
        externBool.append(i)

xSmooth = np.linspace(data[0][0][0], data[0][0][-1], num=100)

'''
data[i][j] is the waveform of the (i+1)th trial, (j+1)th channel
'''

dataUse, errUse, fitUse, fitErrUse, chisqUse, dofUse = [], [], [], [], [], []
for i, e in enumerate(data):
    if i in externBool:
        dataUse.append(data[i])
        errUse.append(err[i])
        fitUse.append([pf[0][i], pf[1][i]])
        fitErrUse.append([pferr[0][i], pf[1][i]])
        chisqUse.append([chisq[0][i], chisq[1][i]] )
        dofUse.append([dof[0][i], dof[1][i]])


plt.close()
fig, ax = plt.subplots( figsize = (12, 12), nrows = 2, ncols = 2)

fig.suptitle('Interference fringes with no external polarizing filter')

for i, axis in enumerate(fig.axes):
    axis.errorbar(dataUse[i][0], dataUse[i][2], errUse[i][1], label='ch1&2')
    axis.errorbar(dataUse[i][0], dataUse[i][3], errUse[i][2], label='ch1&3')
    axis.plot(xSmooth, sinusoid(fitUse[i][0], xSmooth), label = 'ch1&2 fit')
    axis.plot(xSmooth, sinusoid(fitUse[i][1], xSmooth), label = 'ch1&3 fit')

plt.show()


## process/plot filtered runs

externBool = []
for i, e in enumerate(angles):
    if e[2] != 'NONE':
        externBool.append(i)

xSmooth = np.linspace(data[0][0][0], data[0][0][-1], num=100)

dataUse, errUse, fitUse, fitErrUse, chisqUse, dofUse, angleUse = [], [], [], [], [], [], []
for i, e in enumerate(data):
    if i in externBool:
        dataUse.append(data[i])
        errUse.append(err[i])
        fitUse.append([pf[0][i][0], pf[1][i][0]])
        fitErrUse.append([pferr[0][i][0], pf[1][i][0]])
        chisqUse.append([chisq[0][i], chisq[1][i]] )
        dofUse.append([dof[0][i], dof[1][i]])
        angleUse.append(angles[i])

fitUse = np.asarray(fitUse)
fitErrUse = np.asarray(fitErrUse)
chisqUse = np.asarray(chisqUse)
dofUse = np.asarray(dofUse)

angleRel = [float(angleUse[i][2]) for i in range(len(angleUse))]


## cut data
#only use last 19 data, because we were more consistent when taking thise ones
angleRel = np.asarray(angleRel[-19:])
fitUse = np.abs(np.asarray(fitUse[-19:]))
fitErrUse = np.asarray(fitErrUse[-19:])
chisqUse = np.asarray(chisqUse[-19:])
dofUse = np.asarray(dofUse[-19:])

## Fit sine to fringe amplitude

yvals = fft.rfft(np.abs(fitUse[:, 0]))
xvals = fft.rfftfreq(19, 10)

plt.close()
plt.plot(xvals, yvals)
plt.show()
##
def sinusoid(p, x):
    return p[0] * np.cos(  p[1] * x + p[2]  ) + p[3]

guess = [100, .1, 0, 100]

pf1, pferr1, chisq, dof = data_fit(guess, sinusoid, angleRel, fitUse[:,0], fitErrUse[:,0])

xSmooth = np.linspace(0, 180, num=100)


##
plt.close()
plt.title('Interference fringe amplitude with external polarizing filter')
plt.xlabel('Relative angle between external filter and manual half-wave plate')

plt.errorbar(angleRel, np.abs(fitUse[:,0]), fitErrUse[:,0], ls='none', capsize = 3, marker = '.')

plt.plot(xSmooth, sinusoid(pf1, xSmooth))

plt.show()





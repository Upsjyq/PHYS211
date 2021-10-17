##preamble

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


##function definition

def expfunc_bg(p, x): #p[1] = \lambda
    return (p[0]*np.exp(-x * p[1])) + p[2]

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



## Data - Na

Emitter= 'Na-22'
peaks = [511, 1270]
numPeaks = len(peaks)

filePath = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Python Scripts/GammaX-Sections/Na-data.txt'

#generate empty lists of appropriate lengths
X = [0 for i in range(numPeaks)]
numPlates = X.copy()
times = X.copy()
G = X.copy()
N = X.copy()

X0, numPlates0, times0,  G0, N0, G1, N1 = np.loadtxt(filePath, unpack=True, skiprows=2)

X[0] = X0
X[1] = X0
numPlates[0] = numPlates0
numPlates[1] = numPlates0
times[0] = times0
times[1] = times0
G[0] = G0
G[1] = G1
N[0] = N0
N[1] = N1

print(G) #random test


dN = [np.sqrt(N[i]) for i in np.arange(numPeaks)]
dT = [np.ones(len(times[i]), dtype='int32') for i in np.arange(numPeaks)]

print(dN)


## Data - Cs

Emitter = 'Cs-137'

N1 = np.array([1732,2307, 1793, 1462, 769])
N2 = np.array([4523, 6450, 7862, 7152, 3628, 1895, 8961])
dN1 = np.sqrt(N1)
dN2 = np.sqrt(N2)


times = np.array([137, 233, 202, 252, 246])
times2 = np.array([137, 202, 252, 246, 207, 211, 282])
dT1 = np.ones(len(times))
dT2 = np.ones(len(times2))


## Data - Ba

Emitter='Ba-133'

N1 = np.array([1732,2307, 1793, 1462, 769])
N2 = np.array([4523, 6450, 7862, 7152, 3628, 1895, 8961])
dN1 = np.sqrt(N1)
dN2 = np.sqrt(N2)


times = np.array([137, 233, 202, 252, 246])
times2 = np.array([137, 202, 252, 246, 207, 211, 282])
dT1 = np.ones(len(times))
dT2 = np.ones(len(times2))


## Processing

R = [N[i]/times[i] for i in range(numPeaks)]
print(R)
#R1 = N1/times1
#R2 = N2/times2

dR = [[] for i in range(numPeaks)]
#dR1 = np.zeros(len(times[0]))

for j in range(numPeaks):
    for i in range(len(times[j])):
        dR[j].append(R[j][i] * np.sqrt( (dT[j][i]/times[j][i])**2 + (dN[j][i]/N[j][i])**2 ))

print(dR)

#dR2 = np.zeros(len(times2))
#for i in range(len(times2)):
#    dR2[i] = R2[i] * np.sqrt( (dT2[i]/times2[i])**2 + (dN2[i]/N2[i])**2 )

## Fitting
guess = [0,0,0]
pf, pferr, chisq, dof = [], [], [], []
for i in range(len(X)):
    pf0, pferr0, chisq0, dof0 = data_fit(guess,expfunc_bg, X[i], R[i], dR[i])
    pf.append(pf0)
    pferr.append(pferr0)
    chisq.append(chisq0)
    dof.append(dof0)


## Plotting
plt.clf()
fig, ax = plt.subplots(figsize = (12,5), nrows = int(len(pf)/2), ncols=2)
#ax1 = fig.add_subplot(1,2,1)
#ax2 = fig.add_subplot(1,2,2)

fig.suptitle('Attenuation of %s Gammas by Al Shielding' %Emitter)

xSmooth = [np.linspace(min(X[i]), max(X[i]), num=100) for i in range(len(X))]

ax[0].plot(xSmooth[0], expfunc_bg(pf[0], xSmooth[0]), 'r--', label='fit')
ax[1].plot(xSmooth[1], expfunc_bg(pf[1], xSmooth[1]), 'r--', label='fit')

ax[0].errorbar(X[0], R[0], dR[0], fmt='k.', ls='none', capsize=3, label='data')
ax[1].errorbar(X[1], R[1], dR[1], fmt='k.', ls='none', capsize=3, label='data')

ax[0].set_title('Gamma Transmission Intensity, Al --E= %d keV' %peaks[0])
ax[0].set_xlabel('Shielding Thickness, x (mm)')
ax[0].set_ylabel('Count Rate, R (counts/s)')
textAnnot = '$R(x) = R_0 e^{-\lambda x} + B$ \n'
textAnnot += '$R_0 = %.0f \pm %.0f$ counts s$^{-1}$ \n' %(pf[0][0], pferr[0][0])
textAnnot += '$\lambda = $%.2e$ \pm $%.2e mm$^{-1}$ \n' %(pf[0][1], pferr[0][1])
textAnnot += '$B = %.0f \pm %.0f$ counts s$^{-1}$ \n' %(pf[0][2], pferr[0][2])
textAnnot += '$\chi^{2} = %f$ \n' %chisq[0]
textAnnot += '$N = %d$ (dof) \n' %dof[0]
textAnnot += '$\chi^{2} \, N^{-1} = %.2f$ \n' %(chisq[0] / dof[0])


ax[1].set_title('Transmission, %d keV' %peaks[1])
ax[1].set_xlabel('Shielding thickness (mm)')
ax[1].set_ylabel('Net tranmission rate (counts/s)')

plt.show()

## Example
fig = plt.figure(figsize = (14*1.2,6*1.2))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

placementArr = np.array([.4, 1.26, .6])
placementArr *= 1.2


#511keV peak
guess = [300., 1/60. , 0]
pf, pferr, chisq, dof = data_fit(guess,expfunc_bg, x, R1, dR1)

X = np.linspace(x.min(), x.max(), 200)
ax1.errorbar(x, R1, dR1, fmt='k.', capsize=3, label='data')
ax1.plot(X, expfunc_bg(pf, X), 'r-', label='Fit')

ax1.set_title('Gamma Transmission Intensity, Al --E=511keV')
ax1.set_xlabel('Thickness, x(cm)')
ax1.set_ylabel('Count Rate, R (counts/s)')
textAnnot = '$R(x) = R_0 e^{-\lambda x} + B$ \n'
textAnnot += '$R_0 = %.0f \pm %.0f$ counts s$^{-1}$ \n' %(pf[0], pferr[0])
textAnnot += '$\lambda = $%.2e$ \pm $%.2e mm$^{-1}$ \n' %(pf[1], pferr[1])
textAnnot += '$B = %.0f \pm %.0f$ counts s$^{-1}$ \n' %(pf[2], pferr[2])
textAnnot += '$\chi^{2} = %f$ \n' %chisq
textAnnot += '$N = %d$ (dof) \n' %dof
textAnnot += '$\chi^{2} \, N^{-1} = %.2f$ \n' %(chisq / dof)


ax1.text(placementArr[0], placementArr[2], textAnnot, transform=ax.transAxes , fontsize=12,verticalalignment='top')
ax1.legend(loc='upper right')


## Scrap paper
count = 1300
print(np.sqrt(count)/count)
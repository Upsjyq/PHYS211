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


#for comparing attenuation coefficients with literature values
coeffs = []
dcoeffs = []
energies = []

## Data - Na

Emitter= 'Na-22'
peaks = [511, 1270]
numPeaks = len(peaks)

filePath = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Python Scripts/GammaX-Sections/Na-data.txt'

imageName = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Python Scripts/GammaX-Sections/Na-attenuation.png'

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

print(N[0]) #random test


dN = [np.sqrt(abs(N[i]) + G[i]) for i in np.arange(numPeaks)]
dT = [np.ones(len(times[i]), dtype='int32') for i in np.arange(numPeaks)]


guess = []
guess.append([30, 1, 1])
guess.append([80, 1, 1])

print(dN[0])



## Data - Cs

Emitter = 'Cs-137'
peaks = [31, 662]
numPeaks = len(peaks)

filePath = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Python Scripts/GammaX-Sections/Cs-data.txt'

imageName = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Python Scripts/GammaX-Sections/Cs-attenuation.png'

#generate empty lists of appropriate lengths
X = [0 for i in range(numPeaks)]
times = X.copy()
G = X.copy()
N = X.copy()


X0, times0, G0, N0, G1, N1 = np.loadtxt(filePath, unpack=True, skiprows=2)

zeros = (np.where(G0 == 0)[0])
length0 = zeros[0]

X[0] = X0[:length0]
X[1] = X0
#print(X[0], X[1])

times[0] = times0[:length0]
times[1] = times0
G[0] = G0[:length0]
G[1] = G1
N[0] = N0[:length0]
N[1] = N1

#print(G) #random test
print(times[0])


dN = [np.sqrt(abs(N[i]) + G[i]) for i in np.arange(numPeaks)]
dT = [np.ones(len(times[i]), dtype='int32') for i in np.arange(numPeaks)]

guess = []
guess.append([80,1,1])
guess.append([30,1,1])

#print(dN)


## Data - Ba

Emitter= 'Ba-133'
peaks = [31, 81, 356]
numPeaks = len(peaks)

filePath = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Python Scripts/GammaX-Sections/Ba-data.txt'

imageName = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Python Scripts/GammaX-Sections/Ba-attenuation.png'

#generate empty lists of appropriate lengths
X = [0 for i in range(numPeaks)]
numPlates = X.copy()
times = X.copy()
G = X.copy()
N = X.copy()


def conv(fld):
    return -float(fld[:-1]) if fld.endswith(b'-') else float(fld)

X0, times0, G0, N0, G1, N1, G2, N2 = np.loadtxt(filePath, unpack=True, skiprows=2)


X = [X0 for i in peaks]
times = [times0 for i in peaks]
G[0] = G0
G[1] = G1
G[2] = G2
N[0] = N0
N[1] = N1
N[2] = N2

print(N)
print(G) #random test


dN = [np.sqrt(abs(N[i]) + G[i]) for i in np.arange(numPeaks)]
dT = [np.ones(len(X[1]), dtype='int32') for i in np.arange(numPeaks)]

print(dN)

guess = []
guess.append([350, 10, 0])
guess.append([120, 10, 0])
guess.append([80, 10, 0])


## Processing

R = [N[i]/times[i] for i in range(numPeaks)]
print(R[0])
#R1 = N1/times1
#R2 = N2/times2

dR = [[] for i in range(numPeaks)]
#dR1 = np.zeros(len(times[0]))

for j in range(numPeaks):
    for i in range(len(times[j])):
        dR[j].append(R[j][i] * np.sqrt( (dT[j][i]/times[j][i])**2 + (dN[j][i]/N[j][i])**2 ))

print(dR[0])

#dR2 = np.zeros(len(times2))
#for i in range(len(times2)):
#    dR2[i] = R2[i] * np.sqrt( (dT2[i]/times2[i])**2 + (dN2[i]/N2[i])**2 )

## Fitting
#guess = [0,0,0]
pf, pferr, chisq, dof = [], [], [], []
for i in range(len(X)):
    pf0, pferr0, chisq0, dof0 = data_fit(guess[i],expfunc_bg, X[i], R[i], dR[i])
    pf.append(pf0)
    pferr.append(pferr0)
    chisq.append(chisq0)
    dof.append(dof0)


#for comparing attenuation coefficients with the literature values

isIncluded = True
for i in peaks:
    if(i not in energies):
        isIncluded = False
        break

if not isIncluded:
    for i in range(len(peaks)):
        energies.append(peaks[i])
        coeffs.append(pf[i][1])
        dcoeffs.append(pferr[i][1])

print(energies)
print(coeffs)

## Plotting
plt.clf()
numCols = 2
numRows = int((len(pf)+1)/numCols)
textPlacement = [[.4, .91], [.4, .91], [.4, .91]]

fig, ax = plt.subplots(figsize = (12,5*numRows), nrows = numRows, ncols=numCols)
if(numPeaks%2 == 1):
    fig.delaxes(ax[numRows-1][1])

fig.suptitle('Attenuation of %s Gammas by Al Shielding' %Emitter)

if(numRows > 1):
    fig.tight_layout(pad=5)

xSmooth = [np.linspace(min(X[i]), max(X[i]), num=100) for i in range(len(X))]


#ax[1].plot(xSmooth[1], expfunc_bg(pf[1], xSmooth[1]), 'r--', label='fit')


#ax[1].errorbar(X[1], R[1], dR[1], fmt='k.', ls='none', capsize=3, label='data')

for i, axis in enumerate(fig.axes):
    print(i)
    if(i >= numPeaks):
        break
    axis.errorbar(X[i], R[i], dR[i], fmt='k.', ls='none', capsize=3, label='data')
    axis.plot(xSmooth[i], expfunc_bg(pf[i], xSmooth[i]), 'r--', label='fit')
    axis.set_title('Gamma Transmission Intensity, Al --E= %d keV' %peaks[i])
    axis.set_xlabel('Shielding Thickness, x (mm)')
    axis.set_ylabel('Count Rate, R (counts/s)')
    textAnnot = '$R(x) = R_0 e^{-\lambda x} + B$ \n'
    textAnnot += '$R_0 = %.1f \pm %.1f$ counts s$^{-1}$ \n' %(pf[i][0], pferr[i][0])
    textAnnot += '$\lambda = $%.2e$ \pm $%.2e mm$^{-1}$ \n' %(pf[i][1], pferr[i][1])
    textAnnot += '$B = %.1f \pm %.1f$ counts s$^{-1}$ \n' %(pf[i][2], pferr[i][2])
    textAnnot += '$\chi^{2} = %.2f$ \n' %chisq[i]
    textAnnot += '$N = %d$ (dof) \n' %dof[i]
    textAnnot += '$\chi^{2} \, N^{-1} = %.2f$ \n' %(chisq[i] / dof[i])
    axis.text(textPlacement[i][0], textPlacement[i][1], textAnnot,
            transform = axis.transAxes, fontsize = 12, verticalalignment='top')
    axis.legend()


#ax[1].set_title('Transmission, %d keV' %peaks[1])
#ax[1].set_xlabel('Shielding thickness (mm)')
#ax[1].set_ylabel('Net tranmission rate (counts/s)')

#plt.savefig(imageName)
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
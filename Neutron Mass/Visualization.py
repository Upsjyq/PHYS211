import numpy as np
import scipy.fft as fft
from scipy import optimize
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple


##function definition

def gaussianBG(p,x):
    return p[0]/(p[2]*np.sqrt(2*np.pi))*np.exp(-(x-p[1])**2/(2*p[2]**2)) + p[3]*x + p[4]
    '''
    p[0] is the total counts in the gaussian
    p[1] is the center of the distribution
    p[2] is the standard deviation
    p[3] is the slope of the background
    p[4] is the y-intercept of the background
    '''

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

## Data Induction & Fitting

ROI = slice(700,900)

dataPath = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Neutron Mass/Raw data/11_12_21'

data1 = np.loadtxt(os.path.join(dataPath, 'beamopen_pb100mm.tsv'), unpack=True, skiprows=27)

data2 = np.loadtxt(os.path.join(dataPath, 'beamopen_pb100mm+fullofparaffin.tsv'), unpack=True, skiprows=27)


xvals = data1[0][ROI]

counts1 = data1[2][ROI]
dCounts1 = np.sqrt(counts1)
rates1 = counts1 / 265.0 #livetime from data tsv
dRates1 = dCounts1/265.0

counts2 = data2[2][ROI]
dCounts2 = np.sqrt(counts2)
rates2 = counts2 / 260.15 #livetime from data tsv
dRates2 = dCounts2/260.15


guess1 = [1100, 800, 30, 0, 100]
guess2 = [2600, 800, 30, 0, 100]

pf1, pferr1, chisq1, dof1 = data_fit(guess1, gaussianBG, xvals, rates1, dRates1)
pf2, pferr2, chisq2, dof2 = data_fit(guess2, gaussianBG, xvals, rates2, dRates2)

## Plotting
plt.close()

fig, ax = plt.subplots(figsize=(14,10), nrows=2, ncols=1)
plt.subplots_adjust(right=0.75)


fig.suptitle('2.3 MeV Peak with and without Paraffin')

ax[0].set_title('100mm Lead Shielding', loc='left')
rate = ax[0].errorbar(xvals, rates1, dRates1, marker='.', markersize=1, linewidth=.5, ls='none', color='k')
fit, = ax[0].plot(xvals, gaussianBG(pf1, xvals), color='b')
bg, = ax[0].plot(xvals, pf1[3] * xvals + pf1[4], color='gray', ls='--')
ax[0].set_ylabel('Counts s$^{-1}$')

text=r'Fit: $\frac{R}{\sigma \sqrt{2 \pi}} \exp (\frac{{-(x-\mu)^2}}{{2\sigma}} ) + mx + b$ '
text += '\n$R = 4.22 \pm 0.54$ counts s$^{-1}$ \n'
text += '$\mu = 805 \pm 2$ channel \n'
text += '$\sigma = 20.7 \pm 2.5$ channel \n'
text += r'$m = -6.8 \pm .4 \times 10^{-4}$ counts s$^{-1}$ channel$^{-1}$'
text += '\n $b = 0.84 \pm 0.04$ counts'
text += '\n \n'
text += 'reduced $\chi^2 = 1.14$'

ax[0].text(1.02, .4, text, transform=ax[0].transAxes)


ax[1].set_title('100mm Lead Shielding + Paraffin Blocks', loc='left')
ax[1].errorbar(xvals, rates2, dRates2, marker='.', markersize=1, linewidth=.5,ls='none', color='k')
ax[1].plot(xvals, gaussianBG(pf2, xvals), color='b')
ax[1].plot(xvals, pf2[3] * xvals + pf2[4], color='gray', ls='--')
ax[1].set_xlabel('channel number')
ax[1].set_ylabel('Counts s$^{-1}$')


text = '\n$R = 12.65 \pm 0.49$ counts s$^{-1}$ \n'
text += '$\mu = 808.1 \pm 0.7$ channel \n'
text += '$\sigma = 19.8 \pm 0.7$ channel \n'
text += r'$m = -7.0 \pm .4 \times 10^{-4}$ counts s$^{-1}$ channel$^{-1}$'
text += '\n $b = 0.77 \pm 0.03$ counts'
text += '\n \n'
text += 'reduced $\chi^2 = 0.90$'

ax[1].text(1.02, .4, text, transform=ax[1].transAxes)

ax[0].legend( [rate, fit, bg], ['photon count rate', 'gaussian fit', 'linear background'], handler_map={tuple: HandlerTuple(ndivide=None)}, bbox_to_anchor = (0, 1.02,1,0.2), borderaxespad=0, loc = 'lower right', ncol=3)

plt.savefig('C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Neutron Mass/Plot Output/GaussianLeadParaffinVsNone.png')

plt.show()


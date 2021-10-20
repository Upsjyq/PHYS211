##preamble

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


##function definition

def expfunc_bg(p, x): #p[1] = \lambda
    return (p[0]*np.exp(-x * p[1])) + p[2]

def residual(p,func, xvar, yvar, err):
    return (func(p, xvar) - yvar)/err

def gaussianBG(p,x):
    return p[0]/(p[2]*np.sqrt(2*np.pi))*np.exp(-(x-p[1])**2/(2*p[2]**2)) + p[3]*x + p[4]


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


## Data Entry

dataFile = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Python Scripts/GammaX-Sections/Na, 0cm.tsv'

channel, count = np.loadtxt(dataFile, skiprows=24, unpack=True)
print(count[:10]) #sanity check
X = channel
N = count
dN = np.sqrt(count)
print(dN[:10])

## Fitting Gaussian

ROI = [350, 425]
XRestr = X[ROI[0]:ROI[1]]
NRestr = N[ROI[0]:ROI[1]]
dNRestr = dN[ROI[0]:ROI[1]]


guess = [18000, 380, 12, 0, 0]

pf, pferr, chisq, dof = data_fit(guess, gaussianBG, XRestr, NRestr, dNRestr)

## Plotting - Spectrum
plt.clf()

xSmooth = np.linspace(ROI[0], ROI[1], 200)

saveFile = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Python Scripts/GammaX-Sections/Na-Spectrum.png'

plt.rcParams["figure.figsize"] = (12,6)
plt.plot(X, N, 'b.', markersize = 2, label='counts')
plt.plot(XRestr, gaussianBG(pf, XRestr), 'k--', linewidth = 1, label='Gaussian Fit, 511keV')

plt.legend()
plt.savefig(saveFile)
plt.show()

## Plotting - Gaussian
plt.clf()

fig = plt.figure(figsize=(12,6))
ax = plt.axes()

saveFile = 'C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Q1/Python Scripts/GammaX-Sections/Na-Spectrum Restricted.png'

xSmooth = np.linspace(ROI[0], ROI[1], 200)
textPlacement = [.62, .77]
textPlacement2 = [.67, .56]

plt.title('Na-22 Spectrum, 511keV peak')
ax.set_xlabel('Channel (arb units; prop. to energy)')
ax.set_ylabel('Count')

ax.errorbar(XRestr, NRestr, dNRestr, fmt = 'b.', linewidth=.5, capsize=0, markersize = 2, label='counts')
ax.plot(XRestr, gaussianBG(pf, XRestr), 'k--', linewidth = .5, label='Gaussian Fit, 511keV')

textAnnot = '$f(x) = A \, (\sigma \sqrt{2\pi})^{-1} $ exp$[-(x - \mu)^2 (2 \sigma^2)^{-1}] \, + bx + c$ \n'
textAnnot+= '$ A = %d \pm %d$ counts \n' %(pf[0], pferr[0])
textAnnot+= '$ \mu = %.2f \pm %.2f $\n' %(pf[1], pferr[1])
textAnnot+= '$ \sigma = %.2f \pm %.2f $\n' %(pf[2], pferr[2])
textAnnot+= '$ b = %.2f \pm %.2f $ counts channel$^{-1}$\n' %(pf[3], pferr[3])
textAnnot+= '$ c = %.2f \pm %.2f $ counts\n' %(pf[4], pferr[4])

textAnnot2 = '$\chi^2 = %.2f$ \n' %(chisq)
textAnnot2+= '$N = %d$ (dof) \n' %(dof)
textAnnot2+= '$\chi^2 / N = %.2f$ \n' %(chisq / dof)

fig.text(textPlacement[0], textPlacement[1], textAnnot,
            transform = fig.transFigure, fontsize = 10, verticalalignment='top')
fig.text(textPlacement2[0], textPlacement2[1], textAnnot2,
            transform = fig.transFigure, fontsize = 10, verticalalignment='top')

ax.legend(loc='upper right')

plt.savefig(saveFile)
plt.show()

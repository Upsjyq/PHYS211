
#%%
import numpy as np
from scipy import optimize
import os

import matplotlib.pyplot as plt


print("Working directory: \n", os.getcwd())


##function definition

def linear(p, x):
    return p[0] + p[1] * x


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


# %% create and visualize raw data arrays

calibScopeVoltages = np.asarray(
    [-14.2, -13.6, -10.6, -4.8, 1.0, 7.0, 13.0]
)
calibMMVoltages = np.asarray(
    [0., .12, .24, .478, .715, .955, 1.2]
)

fDepump = np.asarray(
    [40, 30, 20, 10, 50, 60, 70, 80, 100, 120, 130, 90, 110],
     dtype=float
)

deltaVscopeSmall = np.asarray(
    [2.84, 2.12, 1.48, 0.68, 3.48, 4.2, 4.84, 5.56, 6.96, 8.5, 9.2, 6.2, 7.6]
)

deltaVscopeLarge = np.asarray(
    [4.2, 3.16, 2.12, 1.08, 5.24, 6.24, 7.32, 8.32, 10.4, 12.5, 13.8, 9.4, 11.5]
)

# rawDataFile = 'raw data'
# np.save(rawDataFile, calibScopeVoltages, calibMMVoltages, fDepump, deltaVscopeSmall, deltaVscopeLarge) 

plt.clf()

plt.plot(calibScopeVoltages, calibMMVoltages, ls='none', marker='.')
plt.show()

plt.clf()
plt.plot(fDepump, deltaVscopeSmall, '.')
plt.plot(fDepump, deltaVscopeLarge, '.')
plt.show()

plt.clf()


# %% Voltage calibration fitting

guess = [0, .1]
calibErr = .005 * np.ones(len(calibScopeVoltages) - 1, dtype=float)
calibParams, pferr, chisq, dof = data_fit(guess, linear, calibScopeVoltages[1:], calibMMVoltages[1:], calibErr)

xsmooth = np.linspace(-15, 15, 200)
plt.clf()
plt.plot(calibScopeVoltages, calibMMVoltages, ls='none', marker='.',
    label="calibration data", color='k')
plt.plot(xsmooth, linear(calibParams, xsmooth), 'b--', 
    label="calibration linear fit", linewidth=1)
plt.title("Scope Calibration data")
plt.legend(loc='upper left')
plt.ylabel("Voltage Across Field Coils, V")
plt.xlabel("Voltage Measured at Oscilloscope, V")

annot = "Linear fit parameters: \n"
annot += 'slope: 0.0406 $\pm$ 0.0002 \n'
annot += 'intercept: 0.672 $\pm$ 0.002 \n'
annot += 'reduced $\chi^2$: 0.10'

plt.text(.6, .6, annot, fontsize=10, verticalAlignment='top')

plt.show()

plt.clf()

#%% Linear fitting actual data

guessSmall = [0, 0.07]
guessBig = [0, 0.1]

deltaVErr = 0.08 * np.ones(len(fDepump))

pS, pSerr, chisqS, dofS = data_fit(guessSmall, linear, fDepump, deltaVscopeSmall, deltaVErr)
pL, pLerr, chisqL, dofL = data_fit(guessBig, linear, fDepump, deltaVscopeLarge, deltaVErr)

xsmooth=np.linspace(0, fDepump.max(), 200)

plt.clf()

plt.plot(fDepump, deltaVscopeSmall, ls='none', marker='.',
    label="Data series 1", color = 'r')
plt.plot(xsmooth, linear(pS, xsmooth), 'r--',
    label = "Series 1 linear fit", linewidth=1)
plt.plot(fDepump, deltaVscopeLarge, ls='none', marker='.',
    label='Data series 2', color='b')
plt.plot(xsmooth, linear(pL, xsmooth), 'b--',
    label='Series 2 linear fit', linewidth=1)

plt.title("Voltage Difference Between Depumping Events")
plt.xlabel("Depumping Frequency, kHz")
plt.ylabel("Oscilloscope Voltage Difference, V")
plt.legend(loc='upper left')

plt.show()
plt.clf()

#%% Auxiliary calculations

print("Other calculations and conversions: \n")

print("Small scope voltage per kHz depumping: %f +- %f"%(pS[1], pSerr[1]))
print("Big scope voltage per kHz depumping: %f +- %f\n"%(pL[1], pLerr[1]))

print("zero-field event is observed at Vscope = %f V" %(-3.5))
print('corresponding coil voltage: %f V\n'%( linear(calibParams, -3.5) ))

print('Vertical zero-B-field: B = (4/5)^(3/2) mu n I / r ')
print('\t B = %e * I '%( (4/5) **1.5 * 1.257e-6 * 20 / .1171 ))
print('\t B = %e \n'%( (4/5) **1.5 * 1.257e-6 * 20 / .1171 * .29 ))

print('Horiontal zero-B-field: ')
print('\t B = %e * V / R' %( (4/5) **1.5 * 1.257e-6 * 11 / .1641 ))
print('\t B = %e \n'%( (4/5) **1.5 * 1.257e-6 * 11 / .1641 * linear(calibParams, -3.5) / 1.1 ))

h = 6.626e-34
muB = 5.7883e-9 #eV / G
mu0 = 1.256e-6
r = .1641
R = 1.1

# convert muB to SI base units:
muB = muB * 1.60218e-19 * 10000 # * J/eV * G/T

#factor of 1000 for conversion from kHz
coef = 1000 * h/muB / ( (4/5)**1.5 * mu0 * 11 * calibParams[1] / (r* R) ) 
g1 = coef/pS[1]
err1 = coef /pS[1] * np.sqrt( (pSerr[1]/pS[1])**2 + (.1/1.1)**2 )
g2 = coef/pL[1]
err2 = coef/pL[1] * np.sqrt( (pLerr[1]/pL[1])**2 + (.1/1.1)**2 )

print('Experimentally determined g-factors:')
print('\t g = (h/muB) [ (4/5)^(3/2) mu0 n M / (r R) ]^-1 [Vcoil/f]^-1')
print('\t g = %e [Vcoil/f]^-1'%coef)
print('\t g1 = %f +- %f'%( g1 , err1 ))
print('\t g2 = %f +- %f\n'%( g2, err2 ))

gj = (1 + (.5*(.5+1) + .5*(.5 + 1) + 0 )/(2 * .5*(.5+1))  )
g85 = np.abs( gj * ( 2*(2+1) + .5*(.5+1) - 2.5*(2.5+1) )/(2 * 2 * (2+1)) )
g87 = (gj * ( 2*(2+1) + .5*(.5+1) - 1.5*(1.5+1) )/(2 * 2 * (2+1)) )
print('Literature Lande g-factors:')
print('both isotopes: gj = %f'%gj)
print('Rb-85: gf = %f'%( g85  ))
print('Rb-87: gf = %f\n'%( g87  ))

print('we conclude that g1 corresponds to Rb-87 with t = %f'%( (g87 - g1)/err1) ) 
print('and g2 ~ Rb-85 with t = %f'%( (g85-g2)/err2) )


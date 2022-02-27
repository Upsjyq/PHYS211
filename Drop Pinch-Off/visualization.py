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

plt.rcParams['figure.figsize'] = (10,7)

#%% Induct first data set ("trial 3")

data = np.loadtxt("Trial 3 composite.tsv", skiprows=1)
# print(data)

widths = np.flip(data[1180:2314])
frames = np.arange(len(widths))
widthErr = np.ones(len(widths))

widths= widths[2:]
frames = frames[2:]
widthErr = widthErr[2:]

logWidths = np.log10(widths)
logFrames = np.log10(frames)
logErr = widthErr/(widths * np.log(10))


plt.clf()

plt.plot(logFrames, logWidths)
plt.show()

roi1 = slice(65, -800)
roi2 = slice(15)
roi3 = slice(17, 60)

print('\n\n region a:')
fit1, fitErr1, chisq1, dof1 = data_fit([0,1], linear, logFrames[roi1], logWidths[roi1], logErr[roi1])
print('\n\n region b:')
fit2, fitErr2, chisq2, dof2 = data_fit([0,1], linear, logFrames[roi2], logWidths[roi2], logErr[roi2])
print('\n\n region c:')
fit3, fitErr3, chisq3, dof3 = data_fit([0,1], linear, logFrames[roi3], logWidths[roi3], logErr[roi3])

plt.close()

plt.errorbar(logFrames, logWidths, logErr, ls='none', color='r',elinewidth=0.5, capsize=0 )
plt.scatter(logFrames, logWidths, s=2, color='r', label='data')
plt.plot(logFrames[roi1], linear(fit1, logFrames[roi1]), 'k--', label='linear fit, region (a)')
plt.plot(logFrames[roi2], linear(fit2, logFrames[roi2]), color='k', ls='dotted', label='linear fit, region (b)')
plt.plot(logFrames[roi3], linear(fit3, logFrames[roi3]), 'k-.', label='linear fit, region (c)')

# plt.axvline(logFrames[0]-.05, color='gray', linewidth=1)
plt.axvline(logFrames[16], color='gray', linewidth=1)
plt.axvline(logFrames[61], color='gray', linewidth=1)
plt.axvline(logFrames[-805], color='gray', linewidth=1)

plt.title("Sample 1 Neck Radius vs Frames; Log-Log")
plt.xlabel("Log10 (Frames to Pinch-Off)")
plt.ylabel("Log10 (Neck Radius, px)")
plt.legend(loc='lower right')

plt.show()

#%% Regions 1&2 sample 2

data = np.loadtxt("trial4_data_composite.tsv", skiprows=1)
print(data)
print(len(data))

frames = np.arange(len(widths))
widthErr = np.ones(len(widths))

widths= widths[2:]
frames = frames[2:]
widthErr = widthErr[2:]

logWidths = np.log10(widths)
logFrames = np.log10(frames)
logErr = widthErr/(widths * np.log(10))




roi1 = slice(65,-900)
roi2 = slice(60)

print('\n\n region a:')
fit1, fitErr1, chisq1, dof1 = data_fit([0,1], linear, logFrames[roi1], logWidths[roi1], logErr[roi1])
print('\n\n region b:')
fit2, fitErr2, chisq2, dof2 = data_fit([0,1], linear, logFrames[roi2], logWidths[roi2], logErr[roi2])

plt.close()

plt.errorbar(logFrames, logWidths, logErr, ls='none', color='r',elinewidth=0.5, capsize=0 )
plt.scatter(logFrames, logWidths, s=2, color='r', label='data')
plt.plot(logFrames[roi1], linear(fit1, logFrames[roi1]), 'k--', label='linear fit, region (a)')
plt.plot(logFrames[roi2], linear(fit2, logFrames[roi2]), color='k', ls='dotted', label='linear fit, region (b)')

# plt.axvline(logFrames[0]-.05, color='gray', linewidth=1)
plt.axvline(logFrames[61], color='gray', linewidth=1)
plt.axvline(logFrames[-895], color='gray', linewidth=1)

plt.title("Sample 2 Neck Radius vs Frames; Log-Log")
plt.xlabel("Log10 (Frames to Pinch-Off)")
plt.ylabel("Log10 (Neck Radius, px)")
plt.legend(loc='lower right')

plt.show()



#%% region 3 sample

data = np.loadtxt("region3Data.tsv", skiprows=1)
data = data.T[1]
# print(data)

widths = np.flip(data[0:18])
frames = np.arange(len(widths))
widthErr = 2 * np.ones(len(widths))

widths= widths[-16:]
frames = frames[-16:]
widthErr = widthErr[-16:]

logWidths = np.log10(widths)
logFrames = np.log10(frames)
logErr = widthErr/(widths * np.log(10))

pf, pferr, chisq, dof = data_fit([1,0], linear, logFrames, logWidths, logErr)

plt.close()

plt.errorbar(logFrames, logWidths, logErr, ls='none', color='r',elinewidth=1, capsize=2 )
plt.scatter(logFrames, logWidths, s=2, color='r', label='data')
plt.plot(logFrames, linear(pf, logFrames), 'k--', label='linear fit')

plt.title("Region 3 Neck Radius vs Frames, Log-Log")
plt.xlabel("Log10 (Frames to Pinch-Off)")
plt.ylabel("Log10 (Neck Radius, px)")
plt.legend(loc='lower right')

plt.show()



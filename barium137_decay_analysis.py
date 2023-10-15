#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:32:21 2023

@author: nicoleevans
"""
#step 1: importing required functions and modules

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#step 2: Define model functions

def lin(x, a, b) -> (float):
    return x/(-a) + b

def nonlin(x, a, b) -> (float):
    return b*np.exp(x/(-a))

#step 3: load data
numb, N_b = np.loadtxt('/Users/nicoleevans/phy224/2023_10_04_pm_background.txt', skiprows=2, unpack=True)
nums, N_t = np.loadtxt('/Users/nicoleevans/phy224/2023_10_04_pm_sample.txt', skiprows=2, unpack=True)

N_b.astype(float)
N_t.astype(float)
numb.astype(float)

#step 4: Subtract mean background radiation from data:
    
N_b_mean = np.average(N_b) #average background radiation)
N_s = np.array(N_t-N_b_mean) #create new array of final sample values

#step 5: Calculate uncertainty for each data point

u_N_s = np.array(np.sqrt(N_t-N_b_mean)) #creating array of uncertainty values for N_s using equation 5 in Appendix B

#step 6: Convert count data into rates

delta_t = 5
R = np.array(N_s/delta_t) #creating array of rates (using equation 1 in Appendix B)
u_R = np.array(u_N_s/delta_t) #creating array of rate uncertainties (using equation 1 in Appendix B)

#step 7: Linear regression

z = np.array(np.log(R)) # creating a set of y values where y = ln(R)
time = np.array(numb*delta_t)
u_z = np.array(np.abs(u_R/R)) #creating array of propagated uncertainties (using equation 6 in Appendix B)

popt, pcov = curve_fit(lin, time, z, sigma=u_z, absolute_sigma=True)
lin_pstd = np.sqrt(np.diag(pcov)) #standard deviation
tau_lin = popt[0]
z_nought = popt[1]


#Step 8: Nonlinear regression

popt, pcov = curve_fit(nonlin, time, R, sigma=u_R, absolute_sigma=True)
nonlin_pstd = np.sqrt(np.diag(pcov)) #standard deviation
tau_nonlin = popt[0]
y_nought = popt[1]

#step 9: calculating half-life

half_life_lin = (-tau_lin*(np.log(0.5)))#converting tau into half-life
half_life_nonlin = (-tau_nonlin*(np.log(0.5)))

#step 10: plotting

lin_pred = nonlin(time, tau_lin, np.exp(z_nought)) #creating y-data using parameters predicted by linear-regression
nonlin_pred = nonlin(time, tau_nonlin, y_nought)#creating y-data using parameters predicted by nonlinear-regression
theoretical_tau = -(2.6*60)/np.log(0.5) #converting theoretical half-life into theoretical tau

plt.figure(1) #not ln-scaled
plt.ylabel('$Rate \\ (s^{-1})$')
plt.xlabel('$Time\\ (s)$')
plt.title('Barium-137m decay by gamma emission')
plt.grid(visible=True, which='both', axis='both')
plt.plot(time, lin_pred,color='blue', linestyle='dashed',label='Linear prediction')
plt.errorbar(time, lin_pred, u_z, marker='|', ecolor='blue', capsize=8, fmt = 'none', mfc='blue', mec='blue', ms=10, mew=2, label='Linear prediction uncertainty')
plt.plot(time, nonlin_pred,color='red', linestyle='dashdot', label='Nonlinear prediction')
plt.errorbar(time, nonlin_pred, u_R, marker='|', capsize=5, ecolor='red',fmt = 'none', mfc='red', mec='r', ms=10, mew=2, label='Nonlinear prediction uncertainty')
plt.plot(time, nonlin(time, theoretical_tau, y_nought),color='green', linestyle='solid',label='Theoretical prediction')
plt.legend(loc='upper right', frameon=True)
plt.savefig('Barium137_decay.png')

plt.figure(2) #log-scaled (natural log)
plt.ylabel('$ln(Rate) \\ (s^{-1})$')
plt.xlabel('$Time\\ (s)$')
plt.title('Barium-137m decay by gamma emission, scaled by natural logarithm')
plt.grid(visible=True, which='major', axis='both')
plt.plot(time, lin_pred,color='blue', linestyle='dashed',label='Linear prediction')
plt.errorbar(time, lin_pred, u_z, marker='|', capsize=8,ecolor='blue', fmt= 'none', mfc='blue', mec='blue', ms=10, mew=2, label='Linear prediction uncertainty')
plt.plot(time, nonlin_pred,color='red', linestyle='dashdot',label='Nonlinear prediction')
plt.errorbar(time, nonlin_pred, u_R, marker='|', capsize=5, ecolor='red', fmt='none', mfc='red', mec='r', ms=8, mew=1, label='Nonlinear prediction uncertainty')
plt.plot(time, nonlin(time, theoretical_tau, y_nought),color='green', linestyle='solid',label='Theoretical prediction')
plt.legend(loc='upper right', frameon=True)
plt.yscale('log')
plt.savefig('Barium137_decay_logscaled.png')

plt.show()

#step 11: variance of parameters
u_lin_tau = lin_pstd[0] #linear regression tau uncertainty
u_lin_halflife = u_lin_tau/(half_life_lin**2)
u_nonlin_tau = nonlin_pstd[0] #nonlinear regression tau uncertainty
u_nonlin_halflife = u_nonlin_tau/(half_life_nonlin**2)
print('Linear regression yields a half-life of',half_life_lin/60,'minutes with an uncertainty of +-', u_lin_halflife,'minutes.')
print('Nonlinear regression yields a half-life of',half_life_nonlin/60,'minutes with an uncertainty of +-', u_nonlin_halflife,'minutes.')
print('The theoretical half-life of Barium-137 is 2.6 minutes.')

#step 12: reduced chi-squared
def chi_squared(mesrd_y:list, est_y:list, sigma:list, num_data:int, parameters:int)->float:
    sum_data = 0
    i=0
    while i<len(mesrd_y):
        sum_data += ((mesrd_y[i]-est_y[i])**2)/(sigma[i]**2)
        i+=1
    return sum_data*(1/(num_data-parameters))


print('The reduced chi-squared value using the parameters calculated from linear regression is:', chi_squared(R, lin_pred, u_R, 60, 2))
print('The reduced chi-squared value using the parameters calculated from non-linear regression is:', chi_squared(R, nonlin_pred, u_R, 60, 2))





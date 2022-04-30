#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt

def log_normal_dist(mu, sigma, x):
    return (1/x * sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*(np.log(x)-mu)**2 * sigma**-2)

def black_body_radiation(wavelength, temp):
    cnst_one = 1.1910439499568096e-16
    cnst_two = 1.4387686699318225e-2
    wavelength = wavelength * 1e-6
    return (cnst_one * (wavelength**5 * (np.exp(cnst_two * (wavelength * temp)**-1) - 1))**-1)

data_x = []
data_y = []
temp = 1250

with open('data_blackbody_radiation_1250.csv', 'r') as myData:
    reader = csv.reader(myData, delimiter = ';')
    for row in reader:
        data_x.append(float(row[0]))
        data_y.append(float(row[1]))
        
x = np.linspace(min(data_x), max(data_x))
y_bbr = black_body_radiation(x, temp)
prprt_data = (max(y_bbr)/max(data_y))

convert_data_y = []
for i in range(len(data_x)):
    convert_data_y.append(data_y[i] * prprt_data)

data_y = convert_data_y
    
#Population Mean:
ln_data_x = []
for i in range(len(data_x)):
    ln_data_x.append(np.log(data_x[i]))
    
PM = (np.transpose(ln_data_x)).dot(data_y) / np.sum(data_y)

#Standard Deviation:
frac_up = 0
for i in range(len(data_x)):
    frac_up += (ln_data_x[i] - PM) **2 * data_y[i]

SD = np.sqrt(frac_up / np.sum(data_y))

#Proportionality ralationship between data and model
y = log_normal_dist(PM, SD, x)
prprt = (max(y_bbr)/max(y))

#R-Square
mean_y = np.mean(data_y);
dis_data_y = 0
dis_model_y = 0
dis_bbr_y = 0

for i in range(len(data_y)):
    dis_data_y += (data_y[i] - mean_y) ** 2
    dis_model_y += (data_y[i] - ( log_normal_dist(PM, SD, data_x[i]) * prprt ) ) ** 2
    dis_bbr_y += (data_y[i] - ( black_body_radiation(data_x[i], temp)) ) ** 2
    
r_square = np.round(1 - (dis_model_y / dis_data_y), decimals = 4)
r_square_bbr = np.round(1 - (dis_bbr_y / dis_data_y), decimals = 4)

y *= prprt

plt.plot(data_x, data_y, 'o')
plt.plot(x, y, '-r', label = 'R-Square: %s' % (r_square))
plt.plot(x, y_bbr, '-b', label = 'R-Square: %s' % (r_square_bbr))

plt.legend(loc = 'upper right')
plt.xlabel(r'wavelength ($\mu m$)')
plt.ylabel('Rediation Intensity')
plt.grid()
plt.show
plt.savefig(str(temp) + '.png', dpi=150)

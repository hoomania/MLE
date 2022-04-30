# -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt

def normal_dist(mu, sigma, x):
    return (1/sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*(x-mu)**2 * sigma**-2)

data_x = []
data_y = []

with open('data_lum_func.csv', 'r') as input_data:
    reader = csv.reader(input_data, delimiter = ';')
    for row in reader:
        data_x.append(float(row[0]))
        data_y.append(float(row[1]))
       
#Population Mean:
PM = (np.transpose(data_x)).dot(data_y) / np.sum(data_y)

#Standard Deviation:
frac_up = 0
for i in range(len(data_x)):
    frac_up += (data_x[i] - PM) **2 * data_y[i]

SD = np.sqrt(frac_up / np.sum(data_y))

#Proportionality ralationship between data and model
x = np.linspace(min(data_x), max(data_x))
y = normal_dist(PM, SD, x)
prprt = (max(data_y)/max(y))

#R-Square
mean_y = np.mean(data_y);
dis_data_y = 0
dis_model_y = 0

for i in range(len(data_y)):
    dis_data_y += (data_y[i] - mean_y) ** 2
    dis_model_y += (data_y[i] - ( normal_dist(PM, SD, data_x[i]) * prprt ) ) ** 2
    
r_square = np.round(1 - (dis_model_y / dis_data_y), decimals = 4)

x = np.linspace(min(data_x), max(data_x))
y *= prprt

plt.plot(data_x, data_y, '-o')
plt.plot(x, y, '-r', label = 'R-Square: %s' % (r_square))
plt.legend(loc = 'upper right')
plt.xlabel('wavelength (nm)')
plt.ylabel('Photopic Response Function')
plt.grid()
plt.show
plt.savefig('lf_2.png', dpi=150)

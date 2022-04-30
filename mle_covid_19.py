# -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt

def normal_dist(mu, sigma, x):
    return (1/sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*(x-mu)**2 * sigma**-2)

main_data_x = []
main_data_y = []

with open('data_covid.csv', 'r') as input_data:
    reader = csv.reader(input_data, delimiter = ';')
    for row in reader:
        main_data_x.append(float(row[0]))
        main_data_y.append(float(row[1]))
   
domains = [[1, 83], [83, 200], [200, 334]]
colors = ['#7fff0055', '#32649688', '#8b000088']
mus = []
sigmas = []
prprts = []
r_square_array = []
for j in range(len(domains)):
    data_x = []
    data_y = []
    
    for i in range(domains[j][0],domains[j][1]):
         data_x.append(main_data_x[i])
         data_y.append(main_data_y[i])
         
    #Population Mean:
    PM = (np.transpose(data_x)).dot(data_y) / np.sum(data_y)
    mus.append(PM)
 
    #Standard Deviation:
    frac_up = 0
    for i in range(len(data_x)):
        frac_up += (data_x[i] - PM) **2 * data_y[i]
 
    SD = np.sqrt(frac_up / np.sum(data_y))
    sigmas.append(SD)
    
    #Proportionality ralationship between data and model
    x = np.linspace(min(data_x), max(data_x))
    y = normal_dist(PM, SD, x)
    prprt = (max(data_y)/max(y))
    prprts.append(prprt)
 
    #R-Square
    meanY = np.mean(data_y)
    dis_data_y = 0
    dist_model_y = 0
 
    for i in range(len(data_y)):
        dis_data_y += (data_y[i] - meanY) ** 2
        dist_model_y += (data_y[i] - ( normal_dist(PM, SD, data_x[i]) * prprt ) ) ** 2
     
    r_square = np.round(1 - (dist_model_y / dis_data_y), decimals = 4)
    r_square_array.append(r_square)
 

x = np.linspace(min(main_data_x), max(main_data_x))
yt = 0
for i in range(len(mus)):
    yt += normal_dist(mus[i], sigmas[i], x) * prprts[i]
    yc = normal_dist(mus[i], sigmas[i], x) * prprts[i]
    plt.plot(x, yc, '-', color= colors[i], label = 'R-Square: %s' % (r_square_array[i]))
    
plt.plot(main_data_x, main_data_y, '-b')
plt.plot(x, yt, '-', color="#000000")
plt.xlabel('Day')
plt.ylabel('Death')
plt.legend(loc = 'upper left')
plt.grid()
plt.show
plt.savefig('covid_t.png', dpi=150)
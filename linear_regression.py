# -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt

data_x = []
data_y = []

with open('data_linear_reg.csv', 'r') as input_data:
    reader = csv.reader(input_data, delimiter = ';')
    for row in reader:
        data_x.append(float(row[0]))
        data_y.append(float(row[1]))
        
A = []
b = []
for x in data_x:
    A.append([x, 1])
    
trps_A = np.transpose(A)
model = np.linalg.inv(np.dot(trps_A, A))
model = np.dot(model, trps_A)
model = np.dot(model, data_y)

#R-Square
mean_y = np.mean(data_y);
dis_data_y = 0
dis_model_y = 0

for i in range(len(data_y)):
    dis_data_y += (data_y[i] - mean_y) ** 2
    dis_model_y += (data_y[i] - ( (data_x[i] * model[0]) + model[1])) ** 2
    
r_square = np.round(1 - (dis_model_y / dis_data_y), decimals = 2)

x = np.linspace(min(data_y), max(data_x))
y = (model[0]*x) + model[1]

plt.plot(data_y, data_y, 'o')
plt.plot(x, y, '-r', label = 'R-Square: %s' % (r_square))
plt.legend(loc = 'upper left')
plt.grid()
plt.show

# Scaling Data:

# For X Distance:

import pandas as pd
import numpy as np
from numpy import savetxt

data = pd.read_csv("ce889_dataCollection.csv")
data = data.dropna(axis=0)
x_dis = data.XDis
l = len(x_dis)
for i in range(0, l, 1):
    x_dis[i] = (x_dis[i] - min(x_dis)) / (max(x_dis) - min(x_dis))

# For Y Distance:

y_dis = data.YDis
l = len(y_dis)
for i in range(0, l, 1):
    y_dis[i] = (y_dis[i] - min(y_dis)) / (max(y_dis) - min(y_dis))

# For Y Velocity:

y_vel = data.VelY
l = len(y_vel)
for i in range(0, l, 1):
    y_vel[i] = (y_vel[i] - min(y_vel)) / (max(y_vel) - min(y_vel))

# For X Velocity:

x_vel = data.VelX
l = len(x_vel)
for i in range(0, l, 1):
    x_vel[i] = (x_vel[i] - min(x_vel)) / (max(x_vel) - min(x_vel))

w = np.array([[0.1], [0.2], [0.3], [0.4]])
ww = np.array([[0.5], [0.6], [0.7], [0.8]])
weights1 = w
weights2 = ww
bias = np.array([[0.3], [0.4]])
lr = 0.8
la = 0.6


def sigmoid(x):
    return 1 / (1 + 2.71828 ** (la * -x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


for epoch in range(1000):
    neuron1 = (x_dis * w[0]) + (y_dis * w[1]) + bias[0]
    neuron2 = (x_dis * w[2]) + (y_dis * w[3]) + bias[0]
    actv1 = sigmoid(neuron1)
    actv2 = sigmoid(neuron2)

    neuron3 = (actv1 * ww[0]) + (actv2 * ww[1]) + bias[1]
    neuron4 = (actv1 * ww[2]) + (actv2 * ww[3]) + bias[1]
    actv3 = sigmoid(neuron3)
    actv4 = sigmoid(neuron4)

    # error for first output
    error1 = actv3 - x_vel
    error2 = actv4 - y_vel

    # Back Propagation
for i in actv3:
    grad_d1 = la * i * (1 - i) * error1

for i in actv4:
    grad_d2 = la * i * (1 - i) * error2

for i in grad_d1:
    ww[0] = lr * i * 0.5
    ww[1] = lr * i * 0.6

for i in grad_d2:
    ww[2] = lr * i * 0.7
    ww[3] = lr * i * 0.8

for i in actv2:
    grad_d3 = la * i * (1 - i) * (actv3 * ww[0] + actv4 * ww[1])

for i in actv1:
    grad_d4 = la * i * (1 - i) * (actv3 * ww[0] + actv4 * ww[1])

for i in grad_d3:
    w[0] = lr * i * 0.5
    w[1] = lr * i * 0.6

for i in grad_d4:
    w[2] = lr * i * 0.7
    w[3] = lr * i * 0.8

for i in w:
    print (i)
for i in ww:
    print (i)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

data = pd.read_csv("ce889_dataCollection.csv")
data = data.dropna(axis=0)


class Feedforward:

    def __init__(self):
        pass

    def normalize(self):
        global x_dis_N, y_dis_N, x_vel_N, y_vel_N

        # Here the data is scaled between 0 and 1, so that the values don't variate largely.
        x_dis = data.XDis.values.reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        xdisscaled = min_max_scaler.fit_transform(x_dis)
        x_dis_N = pd.DataFrame(xdisscaled)
        x_dis_N = x_dis_N.to_numpy()

        y_dis = data.YDis.values.reshape(-1,1)
        ydisscaled = min_max_scaler.fit_transform(y_dis)
        y_dis_N = pd.DataFrame(ydisscaled)
        y_dis_N = y_dis_N.to_numpy()

        xvel = data.VelX.values.reshape(-1, 1)
        xvelscaled = min_max_scaler.fit_transform(xvel)
        x_vel_N = pd.DataFrame(xvelscaled)
        x_vel_N = x_vel_N.to_numpy()

        yvel = data.VelY.values.reshape(-1, 1)
        yvelscaled = min_max_scaler.fit_transform(yvel)
        y_vel_N = pd.DataFrame(yvelscaled)
        y_vel_N = y_vel_N.to_numpy()

    def train_test(self):

        # Data is transformed into training and validation sets to maximize the training on validation.
        global xdis_train,  xdis_valid, ydis_train, ydis_valid
        global xvel_train, xvel_valid, yvel_train, yvel_valid
        xdis_train, xdis_valid, ydis_train, ydis_valid = train_test_split(x_dis_N, y_dis_N, train_size=0.7, test_size=0.3)
        xvel_train, xvel_valid, yvel_train, yvel_valid = train_test_split(x_vel_N, y_vel_N, train_size=0.7, test_size=0.3)

    def para(self):

        # All essentials values are initialized that will be used later on in the code.
        # A FF NN defines a map, while learning values of parameters while iterations.
        global w1, w2, w3, lr, la, mo, mo_up, m, ww1, ww2, ww3
        la = 0.6  # I tried to get hold of L2 Regularization to apply to this rate, but hit and trial was time saving.
        lr = 0.2  # Started with a smaller value, 0.1 and modified by comparing.
        m = 0.1
        w1 = np.array([[0.6], [0.8], [0.7], [0.8]])  # Input to Hidden
        w2 = np.array([[0.5], [0.7], [0.5], [0.9]])  # Hidden to Output
        w3 = np.array([[0.3], [0.4]])  # Biases
        mo = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Initial Values for Momentum
        mo_up = []  # Updated values for Momentum
        ww1 = []  # Updated weights for Input to Hidden
        ww2 = []  # Updated weights for Hidden to Output
        ww3 = []  # Updated Biases

        return w1, w2, w3

    def sigmoid(self, x):
        return 1 / (1 + 2.71828 ** (la * -x))

    def feedforward(self, x, y):
        global n3,n4,actv1,actv2,actv3,actv4,error1,error2,e1,e2,w1,w2,w3,mo

        # Storing values of RMSE to plot graph later between training and validating.
        error1 = []
        error2 = []
        for epoch in range(10):

            # For all values of x and y in the Data Collection CSV
            for i in range(len(x)):

                # Where n1,n2 (Hidden) n3,n4 (Output) neurons
                n1 = (x[i] * w1[0]) + (y[i] * w1[2]) + w3[0]
                n2 = (x[i] * w1[1]) + (y[i] * w1[3]) + w3[0]

                actv1 = self.sigmoid(n1)
                actv2 = self.sigmoid(n2)

                n3 = (actv1 * w2[0]) + (actv2 * w2[2]) + w3[1]
                n4 = (actv1 * w2[1]) + (actv2 * w2[3]) + w3[1]

                actv3 = self.sigmoid(n3)
                actv4 = self.sigmoid(n4)

                # RMSE
                e1 = (((actv3 - xvel_train[i])**2)/(len(x)))**(1/2)
                e2 = (((actv4 - yvel_train[i])**2)/(len(x)))**(1/2)

                error1.append(e1)
                error2.append(e2)

                # Early Stopping using comparison

                if (error1[i] < 0.5) and (error1[i] != error1[i-1]):

                    # Gradients and Back propagation for each weight.
                    grad_d1 = lr * actv3 * (1 - actv3) * e1
                    ww2.append(w2[0] + grad_d1 + (m * mo[0]))
                    mo_up.append(grad_d1)

                    grad_d2 = lr * actv4 * (1 - actv4) * e2
                    ww2.append(w2[1] + grad_d2 + (m * mo[1]))
                    mo_up.append(grad_d2)

                    grad_d3 = lr * actv3 * (1 - actv3) * e1
                    ww2.append(w2[2] + grad_d3 + (m * mo[2]))
                    mo_up.append(grad_d3)

                    grad_d4 = lr * actv4 * (1 - actv4) * e2
                    ww2.append(w2[3] + grad_d4 + (m * mo[3]))
                    mo_up.append(grad_d4)

                    grad_b = lr * e1 + (m * mo[4])
                    ww3.append(w3[0] + grad_b)
                    mo_up.append(grad_b)

                    grad_b1 = lr * e2 + (m * mo[5])
                    ww3.append(w3[1] + grad_b1)
                    mo_up.append(grad_b1)

                    ca1 = (grad_d1 * w2[0]) + (grad_d2 * w2[1])
                    calcu1 = lr * actv1 * (1 - actv1) * ca1
                    ca2 = (grad_d1 * w2[2]) + (grad_d2 * w2[3])
                    calcu2 = lr * actv1 * (1 - actv1) * ca2

                    grad_d5 = lr * calcu1 * x[i] + (m * mo[6])
                    ww1.append(w1[0] + grad_d5)
                    mo_up.append(grad_d5)

                    grad_d6 = lr * calcu2 * x[i] + (m * mo[8])
                    ww1.append(w1[2] + grad_d6)
                    mo_up.append(grad_d6)

                    grad_d7 = lr * calcu1 * y[i] + (m * mo[7])
                    ww1.append(w1[1] + grad_d7)
                    mo_up.append(grad_d7)

                    grad_d8 = lr * calcu2 * y[i] + (m * mo[9])
                    ww1.append(w1[3] + grad_d8)
                    mo_up.append(grad_d8)

                    # Selecting last 4 values from the appended array.
                    w1 = ww1[-4:]
                    w2 = ww2[-4:]
                    w3 = ww3[-4:]
                    # Updating momentum for every weight using respective gradient.
                    mo = mo_up[-10:]

            # Checks RMSE every 10 epochs.
            if epoch % 10 == 0:
                print('Epoch no:', epoch, ' RMSE: ', e1)


    def validate(self, x, y):
        global actv3_valid, actv4_valid, e_valid1, e_valid2

        # using the same formula with updated weights and validation data.
        n1 = (x * w1[0]) + (y * w1[2]) + w3[0]
        n2 = (x * w1[1]) + (y * w1[3]) + w3[0]

        actv1 = 1 / (1 + 2.71828 ** (la * -n1))
        actv2 = 1 / (1 + 2.71828 ** (la * -n2))

        n3 = (actv1 * w2[0]) + (actv2 * w2[1]) + w3[1]
        n4 = (actv1 * w2[2]) + (actv2 * w2[3]) + w3[1]

        actv3_valid = 1 / (1 + 2.71828 ** (la * -n3))
        actv4_valid = 1 / (1 + 2.71828 ** (la * -n4))

        e_valid1 = (((actv3_valid - xvel_valid) ** 2) / (len(x))) ** (1 / 2)
        e_valid2 = (((actv4_valid - yvel_valid) ** 2) / (len(x))) ** (1 / 2)

        return e_valid1, e_valid2


    def display(self):
        for i in w1:
            print(i)
        for i in w2:
            print(i)

    def plotting(self):

        plt.plot(e_valid1, 'o')
        plt.xlabel('No. of Iterations')
        plt.ylabel('RMSE_Validate')
        plt.title('RMSE')
        plt.show()



o1 = Feedforward()
o1.normalize()
o1.train_test()
o1.para()
o1.feedforward(xdis_train, ydis_train)
o1.display()
o1.validate(xdis_valid, ydis_valid)
o1.plotting()







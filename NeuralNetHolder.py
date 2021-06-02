class NeuralNetHolder:

    def __init__(self):
        global w, ww, bias, lr, la
        w = [0.60053195, 0.80050846, 0.70062021, 0.80059297]
        ww = [0.67380928, 0.94172735, 0.67380928, 1.2107923]
        bias = [0.3, 0.4]
        lr = 0.2
        la = 0.6

    def predict(self, input_row):

        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        input_row = [float(i) for i in input_row.split(',')]
        input_row[0] = (input_row[0] - (-524.5503333)) / (43.58032876 - (-524.5503333))
        input_row[1] = (input_row[1] - 65.93842266) / (497.03634581 - 65.93842266)

        n1 = (input_row[0] * w[0]) + (input_row[1] * w[1]) + bias[0]
        n2 = (input_row[0] * w[2]) + (input_row[1] * w[3]) + bias[0]
        actv1 = self.sigmoid(n1)
        actv2 = self.sigmoid(n2)

        n3 = (actv1 * ww[0]) + (actv2 * ww[1]) + bias[1]
        n4 = (actv1 * ww[2]) + (actv2 * ww[3]) + bias[1]
        actv3 = self.sigmoid(n3)
        actv4 = self.sigmoid(n4)

        actv3 = -524.5503333 + actv3 * (43.58032876 - (-524.5503333))
        actv4 = 65.93842266 + actv4 * (497.03634581 - 65.93842266)

        return actv3, actv4

    def sigmoid(self, x):
        return 1 / (1 + 2.71828 ** (la * -x))






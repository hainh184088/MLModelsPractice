import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadData(data_path):
    with open(data_path) as file:
        data = pd.read_csv(file).values
    print(data.dtype)
    n = data.shape[0]
    X = data[:, 0].reshape(-1,1)
    Y = data[:, 1].reshape(-1,1)
    X = np.hstack((np.ones((n, 1)), X))
    w = np.array([0.,1.]).reshape(-1,1)
    
    plt.plot(X, Y, 'ro')
    plt.axis([0, 100, 400, 2000])
    plt.xlabel('square meters')
    plt.ylabel('price')
    plt.show()

    
class linearRegression:
    def __init__(self):
        return
    def lR(self,X_train,Y_traing,alpha):
        numOfIteration = 100
        self.alpha = alpha
        for i in range(1, numOfIteration):
            r = np.dot(X_train, w) - Y_train
            cost[i] = 0.5*np.sum(r*r)
            w[0] -= alpha*np.sum(r)
            # correct the shape dimension
            w[1] -= alpha*np.sum(np.multiply(r, X_train[:,1].reshape(-1,1)))
            print(cost[i])

if __name__ == '__main__':
    loadData(data_path='./data_linear.csv')
    newLinear = linearRegression()
    newLinear.lR(X,Y,0.1)

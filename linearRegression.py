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

    # plt.scatter(x,y)
    plt.plot(X, Y, 'ro')
    plt.axis([0, 100, 400, 2000])
    plt.xlabel('square meters')
    plt.ylabel('price')

    X = np.hstack((np.ones((n, 1)), X))
    w = np.array([0.,1.]).reshape(-1,1)

    return X,Y,w,n

    
class linearRegression:
    def __init__(self):
        return
    def lR(self, X_train,Y_train,alpha):
        numOfIteration = 100
        cost = np.zeros((numOfIteration,1))
        for i in range(1, numOfIteration):
            r = np.dot(X_train, w) - Y_train
            cost[i] = 0.5*np.sum(r*r)
            w[0] -= alpha*np.sum(r)
            # correct the shape dimension
            w[1] -= alpha*np.sum(np.multiply(r, X_train[:,1].reshape(-1,1)))
            print(cost[i])
    def predict(self,X_train,Y_train,w,n):
        predict = np.dot(X_train,w)
        plt.plot((X_train[0][1],X_train[n-1][1]),(predict[0],predict[n-1]),'r')
        plt.show()

if __name__ == '__main__':
    X,Y,w,n = loadData(data_path='./data_linear.csv')
    newLinear = linearRegression()
    newLinear.lR(X,Y,0.0000000000001)
    newLinear.predict(X,Y,w,n)

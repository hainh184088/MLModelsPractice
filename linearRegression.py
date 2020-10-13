import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadData(data_path):
    with open(data_path) as file:
        data = pd.read_csv(file)
    data.dropna(subset=['Births by caesarean section (%)','Births attended by skilled health personnel (%)'], inplace=True)
    X = data['Births by caesarean section (%)'].to_numpy()
    Y = data['Births attended by skilled health personnel (%)'].to_numpy()
    X = X.reshape((X.shape[0], 1))
    Y = Y.reshape((Y.shape[0], 1))
    X = X.T
    Y = Y.T
    # print(X.shape)
    plt.plot(X, Y, 'ro')
    plt.axis([5, 50, 80, 100])
    plt.xlabel('Births by caesarean section (%)')
    plt.ylabel('Births attended by skilled health personnel (%)')
    plt.show()
    one = np.ones((X.shape[0], 1))
    Xbar = np.concatenate((one, X), axis = 0)
    print(Xbar)

    
class linearRegression:
    def _init_(self):
        return
    # def

if __name__ == '__main__':
    loadData(data_path='./data.csv')
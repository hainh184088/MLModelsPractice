import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadData(data_path):
    with open(data_path) as file:
        data = pd.read_csv(file)
    data.dropna(subset=['Births by caesarean section (%)','Births attended by skilled health personnel (%)'], inplace=True)
    X = data['Births by caesarean section (%)'].to_numpy()
    Y = data['Births attended by skilled health personnel (%)'].to_numpy()
    print(Y,X)
    
class linearRegression:
    def _init_(self):
        return
    # def

if __name__ == '__main__':
    loadData(data_path='./data.csv')
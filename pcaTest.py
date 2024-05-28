# perform pca on the data
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
# load the data
def loadData():
    data = pd.read_csv('randomData/data_9.csv', header=None)
    data.dropna(inplace=True)
    return data

# perform pca
def performPCA(data):
    x = StandardScaler().fit_transform(data)
    pca = PCA(n_components=5)
    principalComponents = pca.fit_transform(x)
    print(pca.explained_variance_)
    # print eigenvectors
    print(pca.components_)
    principalDf = pd.DataFrame(data=principalComponents)
    return principalDf

if __name__ == '__main__':
    data = loadData()
    start = time.time()
    principalDf = performPCA(data)
    end = time.time()
    print("Time taken: ", end-start)
    print(principalDf.head())
    # save data to csv
    
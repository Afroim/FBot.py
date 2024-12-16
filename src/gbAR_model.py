import numpy as np
import os
from pathlib import Path
import pandas as pd
from numpy.polynomial.polynomial import Polynomial
import random


def get_file_path(fileName):
    file_name = fileName
    base_path_win = "C:\\Users\\Alik\\Documents\\Project\\FBOT\\PY\\FBot.py\\data\\XAUUSD\\D1\\"
    base_path_linux = "/storage/emulated/0/Documents/Pydroid3/FBot/data/XAUUSD/D1/"
    if os.name == 'nt':  # For Windows
        # Define Windows-specific path
        file_path = Path(base_path_win + file_name)
    else:  # For Linux
        # Define Termux
        file_path = Path(base_path_linux + file_name)
    return file_path
    
    
def autoCorrCoef2(binsec, p):
    if p == 0:
        return 1
    n = len(binsec)
    if p >= n:
        raise ValueError("Order p must be smaller than the length of the sequence.")

    mean_binsec = np.mean(binsec)
    numerator = sum((binsec[i] - mean_binsec) * (binsec[i + p] - mean_binsec) for i in range(n - p))
    denominator = sum((binsec[i] - mean_binsec)**2 for i in range(n))
    
    return numerator / denominator if denominator != 0 else 0
    
def autoCorrCoef(binsec, p):
    mean = np.mean(binsec)

    # Variance
    var = np.var(binsec)

    # Normalized data
    ndata = binsec - mean
    size = len(ndata)
    acorr = np.correlate(ndata, ndata, 'full')[size-1: size+p]
    acorr = acorr / var / len(ndata)
    return acorr

def autoCorrMatrix(binsec, p):
    #autocorrs = [autoCorrCoef(binsec, k) for k in range(0, p)]
    autocorrs = autoCorrCoef(binsec, p)
    # Create the autocorrelation matrix
    autocorr_matrix = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            index = abs(i - j)
            autocorr_matrix[i, j] = round(autocorrs[index], 5)
    
    # Print and return the autocorrelation matrix
    #print("Autocorrelation Matrix:")
#    print(autocorr_matrix)
    
    if np.linalg.det(autocorr_matrix) != 0:
    # Вычисление обратной матрицы
        inverse_matrix = np.linalg.inv(autocorr_matrix)
       # print("Обратная матрица:")
#        print(inverse_matrix)
    else:
        print("Матрица необратима (определитель     равен 0).")
    params = np.dot(inverse_matrix, autocorrs[1:])
    print('Params of models')
    model_coef = list(params) +[1- np.sum(np.abs(params))]
    print(model_coef)
#    coefficients = [1] + [
#        -item for item in params
#    ]
    
    #poly = Polynomial(coefficients)
   # roots = poly.roots()
#    roots_abs = np.abs(roots)
#    print("Roots")
#    print(roots)
#    print(roots_abs)
    disrt = np.abs(model_coef)

    return disrt
    
    
def polynomial_distribution(probabilities, num_samples):
    results = []
    for _ in range(num_samples):
        u = random.uniform(0, 1)
        cumulative_sum = 0
        for index, p in enumerate(probabilities):
            cumulative_sum += p
            if u < cumulative_sum:
                results.append(index)
                break
    return results
    

filename = get_file_path('XAUUSD-D1-DIFF.csv')   
df = pd.read_csv(filename)
binsec = df['negative sign'].values.tolist()
binsec = binsec[-350:-200]
#binsec = binsec
distr = autoCorrMatrix(binsec, 10)
samples = polynomial_distribution(distr, 1)
print(len(distr))
print(samples)

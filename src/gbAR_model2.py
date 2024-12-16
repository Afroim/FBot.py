#pylint:disable=W0621
import random
import os
from pathlib import Path
import pandas as pd
import numpy as np
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
    
# Разделение данных на обучающую и тестовую выборки
def split_data(sequence, alpha):
    split_point = int(len(sequence) * alpha)
    return (sequence[:split_point],             
                sequence[split_point:])
    

# Autocorrelation coefficients
def autoCorrCoef(binsec, p):
    mean = np.mean(binsec)
    # Variance
    var = np.var(binsec)
    # Normalized data
    ndata = binsec - mean
    size = len(ndata)
    acorr = np.correlate(
        ndata, ndata, 'full'
    )[size-1: size+p]
    acorr = acorr / var / len(ndata)
    return acorr
    
# Calculate params of gbAR(p) models
def calcModelParams(binsec, p):
    autocorrs = autoCorrCoef(binsec, p)
    # Create the autocorrelation matrix
    autocorr_matrix = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            index = abs(i - j)
            autocorr_matrix[i, j] = round(autocorrs[index], 5)
   
    if np.linalg.det(autocorr_matrix) == 0:
        print("Матрица необратима (определитель     равен 0).")
        mc = [0]*p + [1]
        return (mc, mc)
        
    # Вычисление обратной матри
    inverse_matrix = np.linalg.inv(
     autocorr_matrix)
    #  Вычисление параматров модели
    params = np.dot(inverse_matrix, autocorrs[1:])
    distr = np.abs(params)
    free_coef = 1. - np.sum(distr)
    model_coef = list(params) + [free_coef]
    probas = list(distr) + [free_coef]

    return (model_coef, probas)


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

# Predict of model dbAR(p)
def predict(coefs, probas, seq, p):
    stage = polynomial_distribution(probas, 1)[0]
    co = coefs[stage]
    if stage == p:
        forecast = random.randint(0 ,1)
    elif co > 0:
        forecast = seq[p  - stage]
    else:
        forecast = 1 - seq[p  - stage]
    return forecast
  
    
def approximation_error(sequence, subSeqSize,p):
    nsize = len(sequence)   
    print(nsize) 
    #steps = size - subSeqSize
    steps = nsize - subSeqSize-1
    mismatches = 0 
    error_counter = 0
    max_error_counter = 0

    for i in range(steps):
        end =  i +subSeqSize
        #print(i,end, len(sequence))
        x_window = sequence[i: end]
        coefs, probas=calcModelParams(
                binsec=x_window,
                p=p)
        approx_value = predict(coefs, probas,             
            x_window, 10)
        if approx_value != sequence[end+1]:
            mismatches += 1
            error_counter += 1
        else:
            max_error_counter = max(max_error_counter, error_counter)
            error_counter = 0
    return (max_error_counter + mismatches / steps)  
    
    
def test1():
    filename = get_file_path('XAUUSD-D1-DIFF.csv')   
    df = pd.read_csv(filename)
    binsec = df['negative sign'].values.tolist()
    binsec1 = binsec[-350:-200]
    coefs, probas=calcModelParams(
        binsec=binsec1,
        p=10)
    print(coefs)
    forecast = predict(coefs, probas, binsec1, 10)
    print("forecast:", forecast, "real", binsec[-200])
    
    
def test2():
    filename = get_file_path('XAUUSD-D1-DIFF.csv')   
    df = pd.read_csv(filename)
    binsec = df['negative sign'].values.tolist()[-1000:]
    subSeqSize = 200
    p = 5
    error = approximation_error(binsec, subSeqSize,p)
    print("Error:" ,error)


def test3():
    filename    =get_file_path('bin_min_relative_change.npy')
    binsec = np.load(filename,         
    allow_pickle=True).tolist()
    subSeqSize = 200
    p = 5
    error = approximation_error(binsec, subSeqSize,p)
    print("Error:" ,error)
    
test3()
    
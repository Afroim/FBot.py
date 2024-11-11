##---
import csv
import os
from pathlib import Path
from tabulate import tabulate
import pandas as pd
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
    return sequence[:split_point], sequence[split_point:]


def windowStat(sequence, m):
    n = len(sequence)
    mismatches = 0 
    steps = n - m
    stats = dict()

    for i in range(steps):
        x_window = sequence[i:i + m]
        stat = stats.get(tuple(x_window), dict())
        stat[sequence[i + m]] = stat.get(sequence[i + m], 0) + 1 
        stats[tuple(x_window)] = stat

    proba_stats = {} 
    for item in stats:
        st_dict = stats[item]
        val_0, val_1 = st_dict.get(0,0), st_dict.get(1, 0)
        zsum = val_0 + val_1
        dst = { 0: val_0/zsum, 1: val_1/zsum }
        proba_stats[item] = dst

    return proba_stats


def predict(stats, key):
    stat = stats.get(tuple(key))
    if stat is None:
        return random.randint(0, 1)
    if stat[0] > stat[1]:
        return 0
    elif stat[0] < stat[1]:
        return 1
    else:
        return random.randint(0, 1)

def predict_error(stats, sequence, m):
    n = len(sequence)
    mismatches = 0 
    steps = n - m
    error_counter = 0
    max_error_counter = 0

    for i in range(steps):
        x_window = sequence[i:i + m]
        approx_value = predict(stats, x_window)
        if approx_value != sequence[i + m]:
            mismatches += 1
            error_counter += 1
        else:
            max_error_counter = max(max_error_counter, error_counter)
            error_counter = 0
    return (max_error_counter + mismatches / steps)

filename = get_file_path('XAUUSD-D1-DIFF.csv') 
df = pd.read_csv(filename)
sequence = df['negative sign'].values.tolist()
# print(sequence)
# Разделение на обучающую и тестовую выборки
alpha = 0.8
m = 8
train_seq, test_seq = split_data(sequence, alpha)
stats = windowStat(train_seq, m)
forcase = predict_error(stats, test_seq, m)
print('forcase', forcase)
# for item in stats:
#     print(item, stats[item])


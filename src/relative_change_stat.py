import numpy as np
from deap import base, creator, tools, algorithms
import random
import csv
import os
from pathlib import Path
from tabulate import tabulate
import pandas as pd


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
    

filename = get_file_path('XAUUSD-D1-DIFF.csv')   
df = pd.read_csv(filename)
# sec = df['negative sign'].values.tolist()

min_relative_change = []
max_relative_change = []
for index, row in df.iterrows():
	sec = row['negative sign']
	print(sec)
	if sec == 0:
		max_limit = (df['High'] - df['Open'])/df['Open']
		min_limit = (df['Open'] - df['Low'])/df['Open']
	else:
		max_limit = (df['Open'] - df['Low'])/df['Open']
		min_limit = (df['High'] - df['Open'])/df['Open']

	min_relative_change.append(min_limit)
	max_relative_change.append(max_limit)

avr_min = np.median(min_relative_change)
avr_max = np.median(max_relative_change)
print(avr_max, avr_min)

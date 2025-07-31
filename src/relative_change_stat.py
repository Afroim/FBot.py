import os
import numpy as np
import pandas as pd

#import random
#import csv
#from pathlib import Path
#from tabulate import tabulate


ORIGINAL = 'original/'
RESULT = 'result/'
LOG = 'log/'
BASE = 'data/XAUUSD/D1/'
SRC_FILE = 'XAUUSD-D1-DIFF.csv'

def get_file_path(fileName, target_path,
    base_path=BASE):
    data_file = base_path + target_path + fileName
    current_dir = os.path.dirname(os.path.    
    abspath(__file__))
    base_dir = os.path.abspath(os.path.
    join(current_dir, '..'))  
    
    # Создаём полный путь к поддиректории
    full_path = os.path.join(base_dir, data_file)
    return full_path
    
    
def getOriginalDF():
    filename = get_file_path('XAUUSD-D1-DIFF.csv', ORIGINAL)   
    df = pd.read_csv(filename)
    return df
    
def bin_min_relative_change():
    df = getOriginalDF()
    #sec = df['negative sign'].values.tolist()

    min_relative_change = []
    max_relative_change = []
    for index, row in df.iterrows():
	    sec = row['negative sign']
	    if sec == 0:
	    	max_limit = (row['High'] - row['Open'])/row['Open']
	    	min_limit = (row['Open'] - row['Low'])/row['Open']
	    else:
	    	max_limit = (row['Open'] - row['Low'])/row['Open']
	    	min_limit = (row['High'] - row['Open'])/row['Open']
	    min_relative_change.append(min_limit)
	    max_relative_change.append(max_limit)
    
    avr_min = np.median(min_relative_change)
    avr_max = np.median(max_relative_change)
    print(avr_max, avr_min)
    
    bin_min_relative_change = [1 if item > avr_min else 0 for item in min_relative_change]
    bin_filename = get_file_path('bin_min_relative_change.npy', ORIGINAL)
    np.save(bin_filename, bin_min_relative_change)
    print('Save bin_min_relative_change')
  
    		
def relative_change():
    df = getOriginalDF()
    #sec = df['negative sign'].values.tolist()

    min_relative_change = []
    close_relative_change = []
    max_relative_change = []
    for index, row in df.iterrows():
        sec = row['negative sign']
        close_ = abs(row['Close'] - row['Open'])/row['Open']
        close_relative_change.append(close_)
        if sec == 0:
            max_limit = (row['High'] - row['Open'])/row['Open']
            min_limit = (row['Open'] - row['Low'])/row['Open']
        else:
            max_limit = (row['Open'] - row['Low'])/row['Open']
            min_limit = (row['High'] - row['Open'])/row['Open']
        min_relative_change.append(min_limit)
        max_relative_change.append(max_limit)
        
    	
    return (max_relative_change, 					
    			min_relative_change, 
    			close_relative_change)
    			
def trend():
	df = getOriginalDF()
	sec = df['negative sign'].values.tolist()
	return sec
    
    

    		
    
    
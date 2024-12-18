import os
import platform
from pathlib import Path
import numpy as np
import joblib as jb
# dump, load


def get_file_path(fileName, 
    sub_path = "data/XAUUSD/D1/"):
    data_file = sub_path + fileName
    current_dir = os.path.dirname(os.path.    
    abspath(__file__))
    base_dir = os.path.abspath(os.path.
    join(current_dir, '..'))  
    
    # Создаём полный путь к поддиректории
    full_path = os.path.join(base_dir, data_file)
    return full_path
    
def save2() :
       #quadric1= np.array([1])
       quadric3 = [1, 0, 1, 1, 1, 0, 1, 0, 0]
       quadric5 = [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]
       quadric7 = [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1]
       
       result_list = []
       #result_list.append(quadric1)
       result_list.append(quadric3)
       result_list.append(quadric5)
       result_list.append(quadric7)
       pop_filename = get_file_path('original/quadricFunc1357_best_population.jb')
       jb.dump(result_list, pop_filename)
       
def save1():  
    quadric21=[1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]
    
    quadric22 = [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
    
    bentFunc64 = [1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    
    result_list = []
    result_list.append(quadric21)
    result_list.append(quadric22)
    result_list.append(bentFunc64)
    pop_filename = get_file_path('original/quadricFunc212264_best_population.npy')
    jb.dump(pop_filename, result_list)


def save3() :
       # m = 3
       quadric3 = [1, 0, 1, 1, 1, 0, 1, 0, 0]
       bentTrio1_3 = [1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0]

       # m = 4
       bentTrio224 = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
       # m= 5
       quadric5 = [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]
       bentTrio1_5 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1]        
       # m = 7
       quadric7 = [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1]
       bentTrio1_7 = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]

       
       result_list = []
       result_list.append(quadric3)
       result_list.append(bentTrio1_3)

       result_list.append(bentTrio224)

       result_list.append(quadric5)
       result_list.append(bentTrio1_5)

       result_list.append(quadric7)
       result_list.append(bentTrio1_7)

       pop_filename = get_file_path('original/transform_007.jb')
       jb.dump(result_list, pop_filename)

    
    
if __name__ == "__main__":
    save3()
    
    
   

# Пример массивов разной длины
#array1 = np.array([1, 2, 3])
#array2 = np.array([4, 5, 6, 7])
#array3 = np.array([8])

# Сохранение массивов
#dump([array1, array2, array3], 'arrays.joblib')

# Загрузка массивов
#loaded_arrays = load('arrays.joblib')

#print(loaded_arrays)  # [array([1, 2, 3]), array([4, 5, 6, 7]), array([8])]

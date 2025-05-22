import numpy as np
import os

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

   
def convertIndividual(asize, new_size): 
     filename = 'result/float_act2_last_pop_w5_5.npy'
     full_filename = get_file_path(filename)
     population1 = np.load(full_filename,   allow_pickle=True).tolist()
     population1 = list({tuple(po) for po in population1})
     new_pop = [list(arr) + [0]*asize for arr in population1]
     filename = f'result/float_act2_last_pop_w5_{new_size}.npy'
     out_filename = get_file_path(filename)
     np.save(out_filename, new_pop)
     
     
if __name__ == "__main__":
    convertIndividual(5,10)
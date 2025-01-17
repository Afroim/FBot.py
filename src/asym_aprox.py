#pylint:disable=E1101
#pylint:disable=W0613
import os
#import platform
#from pathlib import Path
import numpy as np
from deap import base, creator, tools, algorithms
import random
import csv
from tabulate import tabulate
import pandas as pd

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
    
def identity(x):
    return x[-1]
    
    
def identity2(x):
    return 1^x[-1]
    
    
def identity_adv1(x):
    if 1 not in x:
        return 1
    elif 0 not in x:
        return 1
    else:
        return x[-1]
        
        
def identity_adv(x):
    if 0 in x[0:3]:
        return x[-1]
    else:
        return quadricFunc1(x[1:])
    
    
def linearFunc(x):
    return np.bitwise_xor.reduce(x)
        
 
def quadricFunc1(vector):
    return (np.dot(vector[:-2], vector[1:-1]) % 2) ^ vector[-1]


def quadricFunc21(vector):
    return np.dot(vector[:-1], vector[1:]) % 2
    
   
def quadricFunc22(vector):
    return (np.dot(vector[:-1], vector[1:]) % 2) ^ vector[-2] ^ vector[-1]
    
    
def bentFunc62(x):
    return (x[0] & x[1] & x[2] ) ^\
               (x[0] & x[3] ) ^ \
               (x[1] & x[4] ) ^ \
                (x[2] & x[5])
                
                
def bentFunc63(x):
     return (x[0] & x[1] & x[2]) ^ \
     (x[1] & x[3] & x[4] ) ^\
     (x[0] & x[1]) ^\
     (x[0] & x[3]) ^\
     (x[1] & x[5]) ^\
     (x[2] & x[4]) ^\
     (x[3] & x[4])
     
               
def bentFunc64(x):
    return (x[0] & x[1] & x[2]) ^\
                (x[1] & x[3] & x[4]) ^ \
                (x[2] & x[3] & x[5]) ^ \
                (x[0] & x[3]) ^ \
                (x[1] & x[5]) ^\
                (x[2] & x[3]) ^\
                (x[2] & x[4]) ^\
                (x[2] & x[5]) ^ \
                (x[3] & x[4]) ^ \
                (x[3] & x[5] ) 

     
def bentTrio1(x):
     if 0 in x:
         return quadricFunc1(x)
     else:
         return 1 ^ quadricFunc1(x)


def bentTrio22(x):
    if 0 in x:
        return quadricFunc22(x)
    else:
        return 1 ^ quadricFunc22(x) 


def bentTrio13(x):
  if 0 in x[0:3]:
      return quadricFunc1(x)
  else:
      return 1 ^ quadricFunc1(x)
      
      
def bentTrio130(x):
  if 1 not in x:
      return 1
  elif 0 in x[0:3]:
      return quadricFunc1(x)
  else:
      return 1 ^ quadricFunc1(x)
      
      
def bentTrio31(x):  
  if 0 in x[-3:]:
      return quadricFunc1(x)
  else:
      return 1 ^ quadricFunc1(x)
      
  
def bentNotTrio1(x):
     if 1 in x:
         return quadricFunc1(x)
     else:
         return 1 ^ quadricFunc1(x)
         
    
def notBentTrio1(x):
        return 1^ bentTrio1(x)
        
def partTrioBent1(startIndex, firstPolyDegree):
     def partTrio(x):
         y = x[startIndex:]
         if 0 in y[-firstPolyDegree:]:
              return quadricFunc1(y)
         else:
              return 1 ^ quadricFunc1(y)  
              
     return partTrio
     
def partTrioBent2(startIndex):
     def partTrio(x):
         return quadricFunc1(x[startIndex:])         
     return partTrio
        
        

def identity_transform(x, x_size, func, params):
    return func(x)
    
def linear_transform(x, x_size, func, params):
    n = x_size
    A_flat = params
    A = np.reshape(A_flat, (n, n)) 
    # Линейное преобразование Ax
    Ax = np.dot(A, x) % 2
    func_value = func(Ax) 
    return func_value

 
def affine_transform(x, y_size,
     bent_func, params):
    # длина входного вектора
    n = len(x)
    # длина выходного вектора
    m = y_size 
    nm  = n * m
    # Извлекаем подматрицу A 
    A_flat = params[:nm]
    A = np.reshape(A_flat, (m, n))

    # Извлекаем вектор b
    b = params[nm:]

    # Линейное преобразование Ax
    Ax = np.dot(A, x) % 2

    # Добавляем вектор смещения b
    Ax_b = np.bitwise_xor(Ax, b)
    # Вычисляем значение бинарной функции для преобразованных переменных
    result = bent_func(Ax_b) 

    return result
  
  
def adaf_transform(x, y_size, func, params):
    x1 = x[:-1]
    x2 = x[1:]
    lenght = int(len(params)/2)
    params1 = params[:lenght]
    params2 = params[lenght:]
    r1 = affine_transform(x1, y_size, func, params1)
    r2 = affine_transform(x2, y_size, func, params2)
    return r1^r2
    
    
def adaf_transform2(x, y_size, func, params):
    x1 = x[:-2]
    x2 = x[1:-1]
    x3 = x[2:]
    #len1= int(len(params)/3)
    len1 = 30
    len2 = len1  + len1
    len3  = len2 + len1
    params1 = params[:len1]
    params2 = params[len1:len2]
    params3 = params[len2:len3]
    params4 = params[len3:]
    r1 = affine_transform(x1, len(x1), func, params1)
    r2 = affine_transform(x2, len(x2),func , params2)
    r3 = affine_transform(x3, len(x3),func , params3)
    
    w = [r1, r2, r3]
    res  = affine_transform(w, y_size,quadricFunc1, params4)
    
    return res
    
        
# Global Setting
FUNCTOR =  (
affine_transform, bentTrio31
#affine_transform_adv, quadricFunc21
)
FUNC_NAME = FUNCTOR[1].__name__

def approximation_error(sequence, m, part,q_values):
    n = len(sequence)
    mismatches = 0 
    steps = n - m
    transform = FUNCTOR[0]
    func = FUNCTOR[1]
    for i in range(steps):
        x_window = sequence[i:i + m]
        approx_value = transform(x_window, part, func, q_values)
        if approx_value != sequence[i + m]:
            mismatches += 1
  
    #return (np.abs(0.5 - mismatches / steps ))
    return mismatches / steps


# Эвалюционная Функция
def errorFunc(individual, params):
    err = approximation_error(
            params["sequence"],
            params["m"], params["part"], individual)
    return [err]
        
# Генерация хромосомы
def create_individual1(size):
    return [random.randint(0, 1) for _ in range(size)]
    
    
def create_individual(size):
    length = size
    byte_array = os.urandom((length + 7) // 8)
    sequence = []
    for byte in byte_array:
        for i in range(8):
            if len(sequence) < length:
                sequence.append((byte >> i) & 1)
    return sequence
      
# Разделение данных на обучающую и тестовую выборки
def split_data(sequence, alpha):
    split_point = int(len(sequence) * alpha)
    return sequence[:split_point], sequence[split_point:]
    
# Функция для  матрицы преобразования:
def print_matrix(chromosome, n, m):
    nm = n*m
    A_flat = chromosome[:nm]
    matrix = np.reshape(A_flat, (m, n))
    print("Матрица преобразования:")
    for row in matrix:
        print("  ".join(map(str, row)))
    print(chromosome[nm:])

# Функция для запуска генетического алгоритма
def searchBinQuadraticForm(params):
    # Параметрcdы из словаря
    sequence = params['sequence']
    m = params['m']
    part  = params['part']
    ind_size = params['ind_size']
    BitProba = 1./ ind_size
    # BitProba = 0.1;
    pop_size = params['pop_size']
    generations = params['generations']
    cx_prob = params['cx_prob']
    mut_prob = params['mut_prob']
    alpha = params['alpha']
    #period = params['period']
    mu = params['mu']
    lambda_ = params['lambda']
    algo = params['algo']
    mate = params['mate']
    selection = params['selection']
    
    # Разделение на обучающую и тестовую выборки
    train_seq, test_seq = split_data(sequence, alpha)
    #train_seq = test_seq
    
    # Создание начальной популяции
    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Создаем инструменты для генетического алгоритма
    toolbox = base.Toolbox()
    toolbox.register("individual", 
        tools.initIterate, creator.Individual,         
        lambda:create_individual(ind_size))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
         
    if 0 == mate:   
        toolbox.register("mate", tools.cxTwoPoint)
    elif 1 == mate:
        toolbox.register("mate", tools.cxUniform,
        indpb=0.5)
    elif 2 == mate:
         toolbox.register("mate", tools.cxOnePoint)
    # elif 4 == mate:
    #     toolbox.register("mate", tools.cxPartialyMatched)
    
    toolbox.register("mutate", tools.mutFlipBit, indpb=BitProba)

    if 0 == selection: 
        toolbox.register("select", tools.selBest)
    elif 1 == selection: 
        toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Эволюционная функция
    pr = dict()
    pr["sequence"] = train_seq
    pr["m"] = m
    pr["part"] = part
    toolbox.register("evaluate", 
        errorFunc, params=pr)

    # Список для хранения статистики
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    
    # Проверка наличия файла с последней популяцией
    pop_filename = get_file_path(f'result/{FUNC_NAME}_last_pop_w{m}_{ind_size}.npy')
    if os.path.exists(pop_filename):
        population1 = np.load(pop_filename, allow_pickle=True).tolist()
        population1 = list({tuple(po) for po in population1})
        population = [ creator.Individual( pop ) for pop in population1]
        print("Population loaded from file.")
        # Проверка размера загруженной популяции
        if len(population) < pop_size:
            add_size = pop_size - len(population)
            additional_individuals = toolbox.population(n=add_size)
            population.extend(additional_individuals)
            print(f"Added {add_size} individuals to match pop_size.")
    else:
        population = toolbox.population(n=pop_size)
        print("New population created.")

    # Хранилише для хранения лучших хромосом
    hof = tools.HallOfFame(1)

    #--- Алгоритмы эволюции
    if 1 == algo:
        population, logbook = algorithms.\
            eaSimple(
                population, toolbox,
                cxpb=cx_prob,
                mutpb=mut_prob,
                ngen=generations,
                stats=stats,
                halloffame=hof,
                verbose=True
        )
    elif 2 == algo:
        population, logbook = algorithms.\
            eaMuPlusLambda(
             population, toolbox, 
             mu = mu, 
             lambda_ = lambda_,
             cxpb=cx_prob, 
             mutpb=mut_prob,    
             ngen=generations,
             stats=stats,
             halloffame=hof,
             verbose=True)
    
    population.extend(hof)
    pop_set = list({tuple(po) for po in population})
     # Сохранение последней популяции
    np.save(pop_filename, pop_set)
    final_size = len(pop_set)
    print(f"Final population  saved with {final_size} records.")
    # Оценка функции ошибки на тестовой 
    # выборке для лучшего результата
    pr = dict()
    pr["sequence"] = test_seq
    pr["m"] = m
    pr["part"] = part
    test_errors = [errorFunc(x, params=pr)[0]
     for x in pop_set]
    best_ind = np.argmin(test_errors)
    best_individual = pop_set[best_ind]
    test_error = test_errors[best_ind]
    pr = dict()
    pr["sequence"] = train_seq
    pr["m"] = m
    pr["part"] = part
    train_error = errorFunc(best_individual, pr)[0]
    result_file = get_file_path('log/aprox_result.csv')
    
    # Сохранение результатов в CSV файл
    with open(result_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([FUNC_NAME,m,
        train_error,
        stats.compile(population)['min'],
        test_error, alpha, best_individual])

    print(f"Best individual: {best_individual}")
    data = [
   ["Func Name", FUNC_NAME],
    ["Train Error", train_error],
    ["Min of Popupulation",
        stats.compile(population)['min']],
    ["Test Error", test_error]
    ]
    print(tabulate(data, headers="", tablefmt="plain", numalign="right"))

    return best_individual


def test3():
    filename = get_file_path('XAUUSD-D1-DIFF.csv')
    df = pd.read_csv(filename)
    seq = df['negative sign'].values.tolist()
    print(seq)
    
                    
def print_result():
    global FUNCTOR 
    FUNCTOR = (affine_transform, bentFunc64)
    rel_bin_filename = get_file_path('original/bin_min_relative_change.npy')
    seq = np.load(rel_bin_filename, allow_pickle=True).tolist()
    q_values = (0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0)
    print('q_vlues=', len(q_values))
    alpha = 0.8
    train_seq, test_seq = split_data(seq, alpha)
    m = 5
    n = 6
    error_train = approximation_error(train_seq,m,n,q_values)
    print('error >>', error_train )
    error_test = approximation_error(test_seq, m, n,q_values)
    print('error >>', error_train, error_test )
    
def run_test2():
    global FUNCTOR 
    FUNCTOR = (identity_transform, identity_adv)
    rel_bin_filename = get_file_path('original/bin_min_relative_change.npy')
    seq = np.load(rel_bin_filename, allow_pickle=True).tolist()
    q_values = []
    alpha = 0.8
    train_seq, test_seq = split_data(seq, alpha)
    m = 6
    n = 5
    error_train = approximation_error(train_seq,m,n,q_values)
    print('error >>', error_train )
    error_test = approximation_error(test_seq, m, n,q_values)
    print('error >>', error_train, error_test )


def learn():
    global FUNCTOR, FUNC_NAME
    
    FUNCTOR = (adaf_transform2, bentTrio13)
    #FUNCTOR = (affine_transform, bentFunc64) 
    FUNC_NAME ='ADAF_' + FUNCTOR[1].__name__

    rel_bin_filename = get_file_path('original/bin_min_relative_change.npy')
    seq_original = np.load(rel_bin_filename, allow_pickle=True).tolist()
    #seq = seq_original[-1260:]
    seq  = seq_original
    epoch = 0
    cx_probs = [0.5, 0.3, 0.1]
    #mut_probs = [0.5, 0.7, 0.9]
    mates = [0, 1, 2]
    selections = [0, 1]

    while epoch < 14:
        generations = 20
        #pop_size = 200
        
        proba_index = random.randint(0,2)
        cx_prob = cx_probs[proba_index]
        mut_prob = 1 - cx_prob
        mate = mates[random.randint(0,2)] 
        selection = selections[random.randint(0,1)]
        print(f'epoch --> {epoch}')
        print(f'mate={mate}~({cx_prob},{mut_prob}), selection={selection}')

        params = {
            'sequence': seq,  # Бинарная послед.
            'm': 7,  # Размер окна
            'part': 3,
            'ind_size': 102,
            'pop_size': 256,  # Размер популяции
            'generations': generations,  # Кол.поколений
            'cx_prob': cx_prob,  # Вероятность скрещивания
            'mut_prob': mut_prob,  # Вероятность мутации
            'alpha': 0.8,  # Разбиение на выборки
            'mu': 198,
            'lambda': 128,
            'algo': 2, # идекс алгоритма,
            'mate': mate, # индекс функции скрещиванияя
            'selection': selection
        }
        best_chromosome = searchBinQuadraticForm(params)
        inp = params['m']
        out = params["part"]
        print_matrix(best_chromosome, inp, out)
        epoch += 1

   
if __name__ == "__main__":
    # print_result()
    # learn()
    run_test2()


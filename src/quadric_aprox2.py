#pylint:disable=W0603
#pylint:disable=W0613
#pylint:disable=E1101
import os
import platform
from pathlib import Path
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
    return x[0]
    
    
def identity2(x):
    return x[-1]
    
    
def linearFunc(x):
    return np.bitwise_xor.reduce(x)
        
 
def quadricFunc1(vector):
    return (np.dot(vector[:-2], vector[1:-1]) % 2) ^ vector[-1]


def quadricFunc21(vector):
    return np.dot(vector[:-1], vector[1:]) % 2
    
   
def quadricFunc22(vector):
    return (np.dot(vector[:-1], vector[1:]) % 2) ^ vector[-2] ^ vector[-1]
    
    
def bentFunc72(x):
    return (x[0] & x[1] & x[2] ) ^\
               (x[0] & x[3] ) ^ \
               (x[1] & x[4] ) ^ \
                (x[2] & x[5]) ^x[6]
                
                
def bentFunc73(x):
     return (x[0] & x[1] & x[2]) ^ \
     (x[1] & x[3] & x[4] ) ^\
     (x[0] & x[1]) ^\
     (x[0] & x[3]) ^\
     (x[1] & x[5]) ^\
     (x[2] & x[4]) ^\
     (x[3] & x[4]) ^x[6]
     
               
def bentFunc74(x):
    return (x[0] & x[1] & x[2]) ^\
                (x[1] & x[3] & x[4]) ^ \
                (x[2] & x[3] & x[5]) ^ \
                (x[0] & x[3]) ^ \
                (x[1] & x[5]) ^\
                (x[2] & x[3]) ^\
                (x[2] & x[4]) ^\
                (x[2] & x[5]) ^ \
                (x[3] & x[4]) ^ \
                (x[3] & x[5] ) ^x[6]
     
def bentTrio1(x):
     if 0 in x:
         return quadricFunc1(x)
     else:
         return 1 ^ quadricFunc1(x)


def bentTrio21(x):
    if 0 in x:
         return quadricFunc21(x)
    else:
         return 1 ^ quadricFunc21(x)


def bentTrio22(x):
    if 0 in x:
         return quadricFunc22(x)
    else:
         return 1 ^ quadricFunc22(x)

    

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

 
def affine_transform_1(x, x_size,
     bent_func, params):
    n = x_size # Количество переменных
    nn = n * n
    # Извлекаем подматрицу A 
    A_flat = params[:nn]
    A = np.reshape(A_flat, (n, n))

    # Извлекаем вектор b
    b = params[nn:nn + n]

    # Линейное преобразование Ax
    Ax = np.dot(A, x) % 2

    # Добавляем вектор смещения b
    Ax_b =(Ax + b) % 2
    # Вычисляем значение бинарной функции для преобразованных переменных
    result = bent_func(Ax_b) 

    return result
     
    
def affine_transform_2(x, x_size,
     bent_func, params):
    n = x_size # Количество переменных
    nn = n * n
    # Извлекаем подматрицу A 
    A_flat = params[:nn]
    A = np.reshape(A_flat, (n, n))

    # Извлекаем вектор b
    b = params[nn:nn + n]
    
    # Извлекаем вектор c 
    c = params[nn + n:nn + 2*n]
    
    # Извлекаем скаляр d 
    d = params[-1]

    # Линейное преобразование Ax
    Ax = np.dot(A, x) % 2

    # Добавляем вектор смещения b
    Ax_b =(Ax + b) % 2
    # Вычисляем значение Бент функции для преобразованных переменных
    bent_value = bent_func(Ax_b) 

    # Вычисляем скалярное произведение c ⋅ x
    scalar_product = np.dot(c, x) % 2 # c ∘ x 

    # Аффинное преобразование:
    # f(Ax ⊕ b) ⊕ (c ⋅ x) ⊕ d
    result = (bent_value + scalar_product + d) % 2
    return result
    
# Global Setting
FUNCTOR =  (
affine_transform_1, bentTrio21
#affine_transform_adv, quadricFunc21
)
FUNC_NAME = FUNCTOR[1].__name__

# Функция ошибки
def approximation_error2(sequence, m,q_values):
    n = len(sequence)
    mismatches = 0 
    steps = n - m
    error_counter = 0
    max_error_counter = 0
    affineTransform = FUNCTOR[0]
    func = FUNCTOR[1]
    for i in range(steps):
        x_window = sequence[i:i + m]
        approx_value = affineTransform(x_window, m, func, q_values)
        if approx_value != sequence[i + m]:
            mismatches += 1
            error_counter += 1
        else:
            max_error_counter = max(max_error_counter, error_counter)
            error_counter = 0
    return (max_error_counter + mismatches / steps)

def approximation_error(sequence, m, q_values):
    n = len(sequence)
    mismatches = 0 
    steps = n - m
    affineTransform = FUNCTOR[0]
    func = FUNCTOR[1]
    for i in range(steps):
        x_window = sequence[i:i + m]
        approx_value = affineTransform(x_window, m, func, q_values)
        if approx_value != sequence[i + m]:
            mismatches += 1
  
    return ( mismatches / steps )


# Эвалюционная Функция
def errorFunc(individual, sequence, m):
    err = approximation_error(sequence, m, individual)
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
def print_matrix(chromosome, m):
    A_flat = chromosome[:m*m]
    matrix = np.reshape(A_flat, (m, m))
    print("Матрица преобразования:")
    for row in matrix:
        print("  ".join(map(str, row)))

# Функция для запуска генетического алгоритма
def searchBinQuadraticForm(params):
    # Параметрcdы из словаря
    sequence = params['sequence']
    m = params['m']
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
    mate  = params['mate']
    
    # Разделение на обучающую и тестовую выборки
    train_seq, test_seq = split_data(sequence, alpha)
    #train_seq = test_seq
    
    # Создание начальной популяции
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Создаем инструменты для генетического алгоритма
    toolbox = base.Toolbox()
    toolbox.register("individual",
        tools.initIterate, creator.Individual,         
        lambda:create_individual(ind_size))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
         
    if 1 == mate:   
        toolbox.register("mate", tools.cxTwoPoint)
    elif 2 == mate:
        toolbox.register("mate", tools.cxUniform,
        indpb=0.5)
    elif 3 == mate:
        toolbox.register("mate", tools.    
        cxPartialyMatched)
    elif 4 == mate:
         toolbox.register("mate", tools.cxOnePoint)
    #toolbox.register("mate", partial_crossover, start_idx=0, end_idx=m*m)
    toolbox.register("mutate", tools.mutFlipBit, indpb=BitProba)
    #toolbox.register("mutate", tools.mutShuffleIndexes, indpb=BitProba)
    
    toolbox.register("select", tools.selBest)
    
    # Эволюционная функция
    toolbox.register("evaluate", errorFunc, sequence=train_seq, m=m)

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
    test_errors = [errorFunc(x, test_seq, m)[0]
     for x in pop_set]
    best_ind = np.argmin(test_errors)
    best_individual = pop_set[best_ind]
    test_error = test_errors[best_ind]
    train_error = errorFunc(best_individual, train_seq,m)[0]
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

def test1():
    filename = get_file_path('XAUUSD-D1-DIFF.csv')
    df = pd.read_csv(filename)
    seq = df['negative sign'].values.tolist()

    generations = 100
# Пример вызова функции
    params = {
        'sequence': seq ,  # Бинарная послед.
        'm': 6,  # Размер окна
        'pop_size': 200,  # Размер популяции
        'generations': generations,  # Кол. поколений
        'cx_prob': 0.3,  # Вероятность скрещивания
        'mut_prob': 0.5,  # Вероятность мутации
        'alpha': 0.8,  # Разбиение на обучающую и тестовую выборку
       'mu': 100,
       'lambda': 70
       # 'period':  generations //2 # сохранения
    }

    best_chromosome =             searchBinQuadraticForm(params)
    print_matrix(best_chromosome, params['m'])

def test2():
    rel_bin_filename = get_file_path('original/bin_min_relative_change.npy')
    seq = np.load(rel_bin_filename, allow_pickle=True).tolist()
    generations = 20
# Пример вызова функции
    params = {
        'sequence': seq ,  # Бинарная послед.
        'm': 7,  # Размер окна
        'ind_size': 56,
        'pop_size': 200,  # Размер популяции
        'generations': generations,  # Кол.поколений
        'cx_prob': 0.5,  # Вероятность скрещивания
        'mut_prob': 0.5,  # Вероятность мутации
        'alpha': 0.8,  # Разбиение на выборки
       'mu': 80,
       'lambda': 60,
       'algo':  2, # идекс алгоритма,
       'mate': 2 # индекс функции скрещиванияя
    }
    best_chromosome = searchBinQuadraticForm(params)
    print_matrix(best_chromosome, params['m'])


def test6():
    rel_bin_filename = get_file_path('original/bin_min_relative_change.npy')
    seq = np.load(rel_bin_filename, allow_pickle=True).tolist()
    epoch = 0
    cx_probs = [0.5]*9 + [0.3]*5
    mut_probs = [0.5]*9 + [0.7]*5
    mates = [2]*9 + [1]*5
    while epoch < 14:
        print(f'epoch --> {epoch}')
        generations = 20
    # Пример вызова функции
        params = {
            'sequence': seq ,  # Бинарная послед.
            'm': 8,  # Размер окна
            'ind_size': 72,
            'pop_size': 200,  # Размер популяции
            'generations': generations,  # Кол.поколений
            'cx_prob': cx_probs[epoch],  # Вероятность скрещивания
            'mut_prob': mut_probs[epoch],  # Вероятность мутации
            'alpha': 0.8,  # Разбиение на выборки
           'mu': 80,
           'lambda': 60,
           'algo':  2, # идекс алгоритма,
           'mate': mates[epoch] # индекс функции скрещиванияя
        }
        best_chromosome = searchBinQuadraticForm(params)
        print_matrix(best_chromosome, params['m'])
        epoch += 1 


def test3():
    filename = get_file_path('XAUUSD-D1-DIFF.csv')
    df = pd.read_csv(filename)
    seq = df['negative sign'].values.tolist()
    print(seq)
    
   
def test4():
    pop_filename = get_file_path('result/quadricFunc21_last_population.npy')
    population = np.load(pop_filename, allow_pickle=True).tolist()
    pop_set = list({tuple(po) for po in population})
    print(len(pop_set))
    pop_set.sort(key=sum)
    pop_filename = get_file_path('original/quadricFunc21_best_population.npy')
    np.save(pop_filename, pop_set)
    for bp in pop_set:
        print_matrix(bp, 6)
        
        
def test5():
    global FUNCTOR 
    FUNCTOR = (identity_transform, identity)
    rel_bin_filename = get_file_path('original/bin_min_relative_change.npy')
    seq = np.load(rel_bin_filename, allow_pickle=True).tolist()
    alpha = 0.8
    train_seq, test_seq = split_data(seq, alpha)
    m = 1
 
    error_train = approximation_error(train_seq, m, [1])
    error_test = approximation_error(test_seq, m, [1])
    print('error >>', error_train, error_test )

   
if __name__ == "__main__":
    test6()
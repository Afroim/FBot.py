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
    
 
def quadricFunc1(vector):
    return (np.dot(vector[:-2], vector[1:-1]) % 2) ^ vector[-1]


def quadricFunc21(vector):
    return np.dot(vector[:-1], vector[1:]) % 2
    
   
def quadricFunc22(vector):
    return (np.dot(vector[:-1], vector[1:]) % 2) ^ vector[-2] ^ vector[-1]
        
 
NOT_LINE_FUNC =  quadricFunc1
FUNC_NAME = NOT_LINE_FUNC.__name__

def affine_transform(x, x_size, func, params):
    n = x_size
    A_flat = params
    A = np.reshape(A_flat, (n, n)) 
    # Линейное преобразование Ax
    Ax = np.dot(A, x) % 2
    func_value = func(Ax) 
    return func_value

# Функция ошибки
def approximation_error2(sequence, m,q_values):
    n = len(sequence)
    mismatches = 0 
    steps = n - m
    error_counter = 0
    max_error_counter = 0

    for i in range(steps):
        x_window = sequence[i:i + m]
        approx_value = affine_transform(x_window, m, NOT_LINE_FUNC, q_values)
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

    for i in range(steps):
        x_window = sequence[i:i + m]
        approx_value = affine_transform(x_window, m, NOT_LINE_FUNC, q_values)
        if approx_value != sequence[i + m]:
            mismatches += 1
  
    return ( mismatches / steps )


# Эвалюционная Функция
def errorFunc(individual, sequence, m):
    err = approximation_error(sequence, m, individual)
    return [err]
        
# Генерация хромосомы
def create_individual1(m):
    size = m  # Количество коэффициентов
    size = size * size
    return [random.randint(0, 1) for _ in range(size)]
    
  
def create_individual(m):
    length = m
    length = length*length
    byte_array = os.urandom((length + 7) // 8)
    sequence = []
    for byte in byte_array:
        for i in range(8):
            if len(sequence) < length:
                sequence.append((byte >> i) & 1)
    return sequence
    

def selfTransform(size):
    diagonal_matrix = np.diag([1]*size)
    return diagonal_matrix
    
    
# Разделение данных на обучающую и тестовую выборки
def split_data(sequence, alpha):
    split_point = int(len(sequence) * alpha)
    return sequence[:split_point], sequence[split_point:]
    

# Функция для  матрицы преобразования:
def print_matrix(chromosome, m):
    A_flat = chromosome
    matrix = np.reshape(A_flat, (m, m))
    print("Матрица преобразования:")
    for row in matrix:
        print("  ".join(map(str, row)))

# Функция для запуска генетического алгоритма
def searchBinQuadraticForm(params):
    # Параметрcdы из словаря
    sequence = params['sequence']
    m = params['m']
    BitProba = 1./ m*m
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
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: create_individual(m))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
         
    if 1 == mate:   
        toolbox.register("mate", tools.cxTwoPoint)
    elif 2 == mate:
        toolbox.register("mate", tools.cxUniform,
        indpb=0.5)
    elif 3 == mate:
        toolbox.register("mate", tools.    
        cxPartialyMatched)
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
    pop_filename = get_file_path('result/'+FUNC_NAME+'_' + 'last_population.npy')
    hof_filename =get_file_path('result/'+FUNC_NAME+'_'+'last_hof.npy')
    if os.path.exists(pop_filename):
        population1 = np.load(pop_filename, allow_pickle=True).tolist()
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
    
    # Сохранение последней популяции и лучшего результата
    np.save(pop_filename, population)
    np.save(hof_filename, hof)
    print("Final population and best individual saved.")

    # Оценка функции ошибки на тестовой выборке для лучшего результата
    #best_individual = hof[0]
    train_error = errorFunc(hof[0], train_seq,m)[0]
    population.extend(hof)
    # Sesrch best for test in populate
    pop_set = list({tuple(po) for po in population})
    test_errors = [errorFunc(x, test_seq, m)[0] for x in pop_set]
    best_ind = np.argmin(test_errors)
    best_individual = pop_set[best_ind]
    test_error = test_errors[best_ind]
    result_file = get_file_path('result/aprox_result.csv')
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
    generations = 100
# Пример вызова функции
    params = {
        'sequence': seq ,  # Бинарная послед.
        'm': 5,  # Размер окна
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
    best_chromosome =             searchBinQuadraticForm(params)
    print_matrix(best_chromosome, params['m'])


def test3():
    filename = get_file_path('XAUUSD-D1-DIFF.csv')
    df = pd.read_csv(filename)
    seq = df['negative sign'].values.tolist()
    print(seq)
    
   
def test4():
    pop_filename = get_file_path('original/quadricFunc1_last_population.npy')
    population = np.load(pop_filename, allow_pickle=True).tolist()
    pop_set = list({tuple(po) for po in population})
    print(len(pop_set))
    for bp in pop_set:
        print_matrix(bp, 5)

   
if __name__ == "__main__":
    test4()
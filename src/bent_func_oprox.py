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
    base_path_win = "C:\\Users\\Alik\\Documents\\Project\\Lotto\\data\\Pais\\"
    base_path_linux = "/storage/emulated/0/Documents/Pydroid3/FBot/data/XAUUSD/D1/"
    if os.name == 'nt':  # For Windows
        # Define Windows-specific path
        file_path = Path(base_path_win + file_name)
    else:  # For Linux
        # Define Termux
        file_path = Path(base_path_linux + file_name)
    return file_path
    
# бент функции
def bentFunc61(x):
    return (x[0]&x[1])^(x[2]&x[3])^(x[3]&x[4])

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
    return (x[0] & x[1] & x[2] +\
                x[1] & x[3] & x[4]+ \
                x[2] & x[3] & x[5]+ \
                x[0] & x[3] + \
                x[1] & x[5] +\
                x[2] & x[3] +\
                x[2] & x[4] +\
                x[2] & x[5] + \
                x[3] & x[4] + \
                x[3] & x[5] ) % 2
 
def bentFunc82(v):
    return (v[0] & v[1] & v[2]) ^ (v[0] & v[3]) ^ (v[1] & v[4]) ^ (v[2] & v[5]) ^ (v[6] & v[7])
 
BENT_FUNC =  bentFunc62
FUNC_NAME = BENT_FUNC.__name__

def affine_transform(x, x_size, x_size2,
     bent_func, params):
    n = x_size # Количество переменных
    nn = x_size2
    # Извлекаем подматрицу A из объединённого вектора params (размер n x n)
    A_flat = params[:nn]
    A = np.reshape(A_flat, (n, n))  # Преобразуем в матрицу размером n x n

    # Извлекаем вектор b из объединённого вектора params (следующие n элементов)
    b = params[nn:nn + n]
    
    # Извлекаем вектор c из объединённого вектора params (следующие n элементов)
    c = params[nn + n:nn + 2*n]
    
    # Извлекаем скаляр d (последний элемент объединённого вектора params)
    d = params[-1]

    # Линейное преобразование Ax
    Ax = np.dot(A, x) % 2

    # Добавляем вектор смещения b
    Ax_b =(Ax + b) % 2
    # Вычисляем значение Бент функции для преобразованных переменных
    bent_value = bent_func(Ax_b) 

    # Вычисляем скалярное произведение c ⋅ x
    scalar_product = np.dot(c, x) % 2# c ∘ x 

    # Аффинное преобразование:
    # f(Ax ⊕ b) ⊕ (c ⋅ x) ⊕ d
    result = (bent_value + scalar_product + d) % 2
    return result

# Функция ошибки
def approximation_error(sequence, m, mm,                     q_values):
    n = len(sequence)
    mismatches = 0 
    steps = n - m
    for i in range(steps):
        x_window = sequence[i:i + m]
        approx_value = affine_transform(
        x_window, m, mm, BENT_FUNC, q_values)
        if approx_value != sequence[i + m]:
            mismatches += 1
    return mismatches / steps   
    
# Эвалюционная Функция
def errorFunc(individual, sequence, m, mm):
    # Функция для оценки ошибки на обучающей выборке
    err = approximation_error(sequence,
     m, mm, individual)
    return [err]
        
# Генерация хромосомы (коэффициенты для верхней треугольной матрицы)
def create_individual1(m):
    size = m+1  # Количество коэффициентов
    size = size * size
    return [random.randint(0, 1) for _ in range(size)]
    
  
def create_individual(m):
    length = m+1
    length = length*length
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
    # Параметры из словаря
    sequence = params['sequence']
    m = params['m']
    pop_size = params['pop_size']
    generations = params['generations']
    cx_prob = params['cx_prob']
    mut_prob = params['mut_prob']
    alpha = params['alpha']
    #period = params['period']
    mu = params['mu']
    lambda_ = params['lambda']
    
    m1  = m + 1
    BitProba = 1. / m1*m1
    # Разделение на обучающую и тестовую выборки
    train_seq, test_seq = split_data(sequence, alpha)
    
    # Создание начальной популяции
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Создаем инструменты для генетического алгоритма
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: create_individual(m))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=BitProba)
    #toolbox.register("mutate", tools.mutShuffleIndexes, indpb=BitProba)
    
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Эволюционная функция
    toolbox.register("evaluate", errorFunc, sequence=train_seq, m=m, mm=m*m)

    # Список для хранения статистики
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    
    # Проверка наличия файла с последней популяцией
    pop_filename = get_file_path('last_population.npy')
    hof_filename =get_file_path('last_hof.npy')
    if os.path.exists(pop_filename):
        population1 = np.load(pop_filename, allow_pickle=True).tolist()
        population = [ creator.Individual( pop ) for pop in population1]
        print("Population loaded from file.")
        # Проверка инициализации fitness для каждой особи
        #for ind in population:
            #if not hasattr(ind, 'fitness') or not ind.fitness.valid:
                #ind.fitness.values = toolbox.evaluate(ind)  # Пересчитываем fitness
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
    population, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=cx_prob,
        mutpb=mut_prob,
        ngen=generations,
        stats=stats,
        halloffame=hof,
        verbose=True
    )
    
    #population, logbook = algorithms.\
#        eaMuPlusLambda(
#         population, toolbox, 
#         mu = mu, 
#         lambda_ = lambda_,
#         cxpb=cx_prob, 
#         mutpb=mut_prob, 
#         ngen=generations,
#         stats=stats,
#         halloffame=hof,
#         verbose=True)
    
    # Сохранение последней популяции и лучшего результата
    #popToSave = [list(pop) for pop in population]
    np.save(pop_filename, population)
    #hofToSave = [list(hf) for hf in hof]
    np.save(hof_filename, hof)
    print("Final population and best individual saved.")

    # Оценка функции ошибки на тестовой выборке для лучшего результата
    best_individual = hof[0]
    test_error = errorFunc(best_individual, test_seq, m, m*m)
    result_file = get_file_path('aprox_result.csv')
    # Сохранение результатов в CSV файл
    with open(result_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([FUNC_NAME,m,
        stats.compile(population)['min'],
        test_error[0], alpha, best_individual])

    print(f"Best individual: {best_individual}")
    return best_individual

# sec = [1, 0, 1, 1, 0, 1, 0, 1]
# золото
filename = get_file_path('XAUUSD-D1-DIFF.csv')   
df = pd.read_csv(filename)
sec = df['negative sign'].values.tolist()
generations = 100
# Пример вызова функции
params = {
    'sequence': sec ,  # Бинарная послед.
    'm': 6,  # Размер окна
    'pop_size': 20,  # Размер популяции
    'generations': generations,  # Кол. поколений
    'cx_prob': 0.1,  # Вероятность скрещивания
    'mut_prob': 0.9,  # Вероятность мутации
    'alpha': 0.8,  # Разбиение на обучающую и тестовую выборку
   # 'period':  generations //2 # сохранения
   'mu': 100,
   'lambda': 70
}

best_chromosome = searchBinQuadraticForm(params)
#print(f"Best coefficients: {best_chromosome}")
print_matrix(best_chromosome, params['m'])
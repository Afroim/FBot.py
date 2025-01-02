import os
#import platform
#from pathlib import Path
import numpy as np
from deap import base, creator, tools, algorithms
import random
import csv
from tabulate import tabulate
import joblib as jb

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
  

def getMatrixes(fileName):
     matrix = jb.load(fileName)
     return matrix
     
###  Functions
def randomizer(_):
    return random.randint(0, 1)     
     
def identity(value):
    return value[0]
    
def not_identity(value):
    return 1^value[0]
          
def linearFunc(x):
    return np.bitwise_xor.reduce(x)
        
def quadricFunc1(vector):
    return (np.dot(vector[:-2], vector[1:-1]) % 2) ^ vector[-1]

def quadricFunc22(vector):
    return (np.dot(vector[:-1], vector[1:]) % 2) ^ vector[-2] ^ vector[-1]
     
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
    
def notBentTrio13(x):
    return 1 ^ bentTrio13(x)
            
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
                
def notBentFunc64(x):
    return 1^bentFunc64(x)

###  Transform Functions
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
   
    
def affine_transform(x, y_size, func, params):
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
    result = func(Ax_b) 

    return result
    
   
def adaf_transform(x, y_size, func, params):
    x1 = x[:-1]
    x2 = x[1:]
    ln = len(params)/2
    params1 = params[:ln]
    params2 = params[ln:]
    r1 = affine_transform(x1, y_size, func, params1)
    r2 = affine_transform(x2, y_size, func, params2)
    return r1^r2
    
    
# Глобальный словарь с размерами окон
WINDOW_SIZES = [ 5, 5 ]
TRANS_WIN_SIZE = [6, 6]
MAX_WIN_SIZE = max(WINDOW_SIZES)                
MATRIX_REP = [] 
  
FUNCS =  [
    (bentFunc64, affine_transform),
    (notBentFunc64, affine_transform)
]
FUNC_NAME = 'asym_win05'

# Функция ошибки
def approximation_error(sequence, indexes):
        
    max_win_size = 5
    n = len(sequence)
    isize = len(indexes)
    
    idx  = 0
    sidx = indexes[idx]
    end = max_win_size
    wsize = WINDOW_SIZES[sidx] 
    tr_size = TRANS_WIN_SIZE[sidx]  
    start = end - wsize
    funcs = FUNCS[indexes[idx]]
    func = funcs[0]
    transform = funcs[1]
    tr_matrix = MATRIX_REP[sidx]
    mismatches = 0
    while end < n: 
        x_window = sequence[start:end]
        # print(wsize, start, end,
        #          func.__name__, 
        #          transform.__name__)     
        # print(f'Matrix:{len(tr_matrix)}')
        forecast = transform(
            x_window, tr_size,func, tr_matrix)      
        
        if forecast != sequence[end]:
            mismatches += 1
            
        idx +=1 
        sidx = indexes[idx%isize]
        wsize = WINDOW_SIZES[sidx]
        tr_size = TRANS_WIN_SIZE[sidx]
        funcs = FUNCS[sidx]
        func = funcs[0]
        transform = funcs[1]
        tr_matrix = MATRIX_REP[sidx]
        end +=1
        start = end - wsize
        
    return mismatches/idx


def approximation_error1(sequence, wsize, indexes):
    n = len(sequence)
    mismatches = 0 
    steps = n - wsize
    max_index = len(indexes)
    for i in range(steps):
        x_window = sequence[i:i + wsize]
        ind = indexes[i % max_index]
        q_values = MATRIX_REP[ind]
        func = NOT_LINE_FUNC[ind]
        #print(q_values)
        approx_value = affine_transform(x_window, wsize, func , q_values)
        if approx_value != sequence[i + wsize]:
            mismatches += 1
  
    return ( mismatches / steps )


# Эвалюционная Функция
def errorFunc(individual, sequence):
    err = approximation_error(sequence, individual)
    return [err]
        
# Генерация хромосомы
def create_individual(size):
    m = len(WINDOW_SIZES) - 1
    return [random.randint(0, m) for _ in range(size)]
    
 
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
    xsize = params['xsize']
    BitProba = 1./ xsize
    # BitProba = 0.1;
    wsize = params['win_size']
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
    mutate = params['mutate']
    selection  = params['selection']
    
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

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, 
    creator.Individual,
    lambda: create_individual(xsize))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
         
    if 0 == mate:   
        toolbox.register("mate", tools.cxOnePoint)
    elif 1 == mate:
        toolbox.register("mate", tools.cxUniform,
        indpb=0.5)
    elif 2 == mate:
        toolbox.register("mate", tools.cxTwoPoint)
    
    if 0 == mutate:
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=BitProba)
    elif 1 == mutate:
         toolbox.register("mutate", tools.mutUniformInt, low=0,up=len(MATRIX_REP)-1,indpb=BitProba)
    
    if 0 == selection: 
        toolbox.register("select", tools.selBest)
    elif 1 == selection: 
        toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Эволюционная функция
    toolbox.register("evaluate", errorFunc, sequence=train_seq)

    # Список для хранения статистики
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    
    pop_filename = get_file_path(f'result/{FUNC_NAME}_{xsize}_population.npy')
  
    # Проверка наличия файла с последней популяцией
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
    hof = tools.HallOfFame(10)

    #--- Алгоритмы эволюции
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
    test_errors = [errorFunc(x, test_seq)[0] for x in pop_set]
    best_ind = np.argmin(test_errors)
    best_individual = pop_set[best_ind]
    test_error = test_errors[best_ind]
    train_error = errorFunc(best_individual, train_seq)[0]
    result_file = get_file_path('log/aprox_result.csv')
    # Сохранение результатов в CSV файл
    with open(result_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([FUNC_NAME,wsize,
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
     
     global WINDOW_SIZES, FUNCS, MATRIX_REP 
     
     
     WINDOW_SIZES = [1, 2, 3, 3, 4, 5, 5, 7, 7]

     FUNCS =  [
        (identity, identity_transform), # 1
        (identity, identity_transform), # 2

        (quadricFunc1, linear_transform), # 3
        (bentTrio1, affine_transform),    # 3

        (bentTrio22, affine_transform),    # 4

        (quadricFunc1, linear_transform), # 5
        (bentTrio1, affine_transform),    # 5

        (quadricFunc1, linear_transform),  # 7
        (bentTrio1, affine_transform)     # 7
     ]

     MATRIX_REP = [[0], [0]]
     MATRIX_REP.extend(
     getMatrixes(get_file_path('original/transform_007.jb')))
     alpha = 0.8
     rel_bin_filename = get_file_path('original/bin_min_relative_change.npy')
     seq = np.load(rel_bin_filename, allow_pickle=True).tolist()
     train_seq, test_seq = split_data(seq, alpha)
     # indexes = create_individual(100)
     # indexes = [0,0,0,1,1,2,0,2,2,1,0,1,1,0,1,0,1,1,2,0,2,2,1,0,1,1,2,0,2,2,1,0,1,1,0,1,1,2,0]
     indexes = create_individual(5)
     print(indexes)
     error = approximation_error(train_seq, indexes)
     err2 = approximation_error(test_seq, indexes)
     print(' Train error =', error)  
     print(' Test error =', err2)  
     print(indexes)


def learn1():
    global WINDOW_SIZES, FUNCS, MATRIX_REP 
    WINDOW_SIZES = [1, 2, 3, 3, 4, 5, 5, 7, 7]
    FUNCS =  [
        (identity, identity_transform), # 1
        (identity, identity_transform), # 2
        (quadricFunc1, linear_transform), # 3
        (bentTrio1, affine_transform),    # 3
        (bentTrio22, affine_transform),    # 4
        (quadricFunc1, linear_transform), # 5
        (bentTrio1, affine_transform),    # 5
        (quadricFunc1, linear_transform),  # 7
        (bentTrio1, affine_transform)     # 7
    ]

    MATRIX_REP = [[0], [0], [0]]
    MATRIX_REP.extend(getMatrixes(get_file_path('original/transform_007.jb')))

    rel_bin_filename = get_file_path('original/bin_min_relative_change.npy')
    seq = np.load(rel_bin_filename, allow_pickle=True).tolist()
    generations = 3
# Пример вызова функции
    params = {
        'sequence': seq ,  # Бинарная послед.
        'win_size': 0,  # Размер окна
        'xsize': 10, # размер хромосомы
        'pop_size': 256,  # Размер популяции
        'generations': generations,  # Кол.поколений
        'cx_prob': 0.1,  # Вероятность скрещивания
        'mut_prob': 0.9,  # Вероятность мутации
        'alpha': 0.8,  # Разбиение на выборки
       'mu': 128,
       'lambda': 64,
       'algo':  2, # идекс алгоритма,
       'mate': 2, # индекс функции скрещиванияя
       'mutate': 2 # индекс функции мутации
    }
    best_chromosome = searchBinQuadraticForm(params)
    print(best_chromosome)   


def learn2(meta_param_random):
    global WINDOW_SIZES, TRANS_WIN_SIZE
    global FUNCS, MATRIX_REP
    
    WINDOW_SIZES = [5,5]
    MATRIX_REP = [6, 6]
    FUNCS =  [
        (bentFunc64, affine_transform),
        (notBentFunc64, affine_transform)
    ]

    MATRIX_REP = [
        [0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0]        
    ]
    
    rel_bin_filename = get_file_path('original/bin_min_relative_change.npy')
    seq = np.load(rel_bin_filename, allow_pickle=True).tolist()

    epoch = 0
    cx_probs = [0.5, 0.3, 0.1]
    mut_probs = [0.5, 0.7, 0.9]
    mates = [0, 1, 2]
    mutates = [0, 1]
    selections = [0, 1]

    while epoch < 20:

        print(f'epoch --> {epoch}')

        if meta_param_random:
            proba_index = random.randint(0,2)
            cx_prob = cx_probs[proba_index]
            mut_prob = 1 - cx_prob
            mate = mates[random.randint(0,2)] 
            mutate = mutates[random.randint(0,1)]
            selection = selections[random.randint(0,1)]
        else:
            cx_prob = 0.1
            mut_prob = 0.9
            mate = 1
            mutate = 1
            selection = 0

        print(f'mate={mate}~{cx_prob}, mutate={mutate}~{mut_prob}, selection={selection}')

        params = {
            'sequence': seq ,  # Бинарная послед.
            'generations': 10,  # Кол.поколений
            'win_size': 0,  # Размер окна
            'xsize': 80, # размер хромосомы
            'pop_size': 256,  # Размер популяции
            'alpha': 0.8,  # Разбиение на выборки
            'mu': 128,
            'lambda': 64,
            'cx_prob': cx_prob,  # Вероятность скрещивания
            'mut_prob': mut_prob,  # Вероятность мутации    
            'algo':  2, # идекс алгоритма,
            'mate': mate, # индекс функции скрещиванияя
            'mutate': mutate, # индекс функции мутации
            'selection': selection
        }
        best_chromosome = searchBinQuadraticForm(params)
        print(best_chromosome)    
        epoch += 1
        
      
if __name__ == "__main__":
    learn2(meta_param_random=True)
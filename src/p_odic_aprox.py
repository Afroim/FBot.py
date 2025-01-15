#pylint:disable=W0621
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
import math


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
    
    
def inv_identity(x):
    return 1^x[-1]
    
    
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
     
def identityTrio(x):
    return (x[0] & x[1] & x[2]) ^ x[-1]
     
def partTrioBent2(startIndex):
     def partTrio(x):
         return quadricFunc1(x[startIndex:])         
     return partTrio
        
def decimal_to_binary(value, length): 
    sign = int(value < 0) 
    value = abs(value)

    # Генерация двоичных разрядов
    binary_array = np.zeros(length + sign, dtype=int)
    for i in range(length):
        value *= 2
        binary_array[-(i + 1 + sign)] = int(value >= 1)  # Заполнение со старших индексов
        value %= 1

    # Добавляем бит знака в конец массива
    if sign == 1:
        binary_array[-1] = 1

    return binary_array      
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def act_mod_1(Q):
   return int( ((Q % 1)*2) >1 )
   
def act_sign(Q):
    return int(Q > 0)
    
def act_mod1_on(Q, length): 
    value = Q

    for _ in range(length):
        value %= 1
        value *= 2
        
    return int(value > 1)
        

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

 
def p_odic_transform(params, x, coefs ):   
    rsize  = params["rsize"]
    base = params["base"]
    mod = params["mod"]
    func = params["func"]
    decimal_number = np.dot(x, base ** np.arange(len(x)))
    
    # Вычисление значения полинома
    powers = np.arange(len(coefs))
    polynomial_value = np.sum(np.array(coefs) * (decimal_number ** powers)) % mod

    binary_result = np.binary_repr(polynomial_value, width=rsize)[::-1]
    bresult = list(map(int, binary_result))
    result = func(bresult)
    return result
  
  
def float_transform2(params, x, coefs ):   
    rsize  = params["rsize"]
    base = params["base"]
    mod = params["mod"]
    func = params["func"]
   
    decimal_number = np.dot(x[::-1], base ** np.arange(len(x)))
    
    # Вычисление значения полинома
    powers = np.arange(len(coefs))
    polynomial_value = np.sum(np.array(coefs) * (decimal_number ** powers)) % mod
    
    by = decimal_to_binary(polynomial_value, rsize)

    result = func[len(by)%2](by)
    return result
  
  
def float_sign_transform(params, x, coefs ):   
    rsize  = params["rsize"]
    base = params["base"]
    mod = params["mod"]
    func = params["func"]
    x1 = x[1:]
    sign = x[0]
    decimal_number = np.dot(x1[::-1], base ** np.arange(len(x1)))
    if 1 == sign:
        decimal_number = -decimal_number
    # Вычисление значения полинома
    powers = np.arange(len(coefs))
    polynomial_value = np.sum(np.array(coefs) * (decimal_number ** powers)) % mod
    
    by = decimal_to_binary(polynomial_value, rsize)

    result = func[len(by)%2](by)
    return result
    
    
def float_transform(params, x, coefs ):   
    #rsize  = params["rsize"]
    base = params["base"]
    #mod = params["mod"]
    func = params["func"]
    x1 = x
    sign = 0
    decimal_number = np.dot(x1[::-1], base ** np.arange(len(x1)))
    if 1 == sign:
        decimal_number = -decimal_number
    # Вычисление значения полинома
    powers = np.arange(len(coefs))
    polynomial_value = np.sum(np.array(coefs) * (decimal_number ** powers))
    
    #result = int(((polynomial_value %1)* 2) >= 1)
    result =   func(polynomial_value)
    return result
    
    
def double_transform(params, x, coefs ):   
    #rsize  = params["rsize"]
    base = params["base"]
    #mod = params["mod"]
    #func = params["func"]
    x1 = x[0:-1]
    x2 = x[1:]
    sign1 = x1[-1]
    sign2 = x2[-1]
    decimal_number1 = np.dot(x1[::-1], base ** np.arange(len(x1)))
    decimal_number2 = np.dot(x2[::-1], base ** np.arange(len(x2)))
    if 1 == sign1:
        decimal_number1 = -decimal_number1
    if 1 == sign2:
        decimal_number2 = -decimal_number2
    # Вычисление значения полинома
    size = len(coefs)//2
    powers = np.arange(size)
    p_value1 = np.sum(np.array(coefs[:size]) * (decimal_number1 ** powers))
    
    p_value2 = np.sum(np.array(coefs[size:]) * (decimal_number2 ** powers))
    
    p_value = sigmoid(p_value1+ p_value2) 
    result = int((p_value* 2) >= 1)
    return result
    
    
def trigonometric_series(params, x):
     # Нулевой коэффициент
    a0 = params[0]
    n = len(params)
    midpoint = (n - 1) // 2
    # Коэффициенты для cos(nx)
    a_coeffs = np.array(params[1:midpoint + 1])
    # Коэффициенты для sin(nx)
    b_coeffs = np.array(params[midpoint + 1:])  
    # Создаём массив индексов для n
    n_values = np.arange(1, len(a_coeffs) + 1)
    # Вычисление суммы ряда
    cos_terms = a_coeffs * np.cos(n_values * x)
    sin_terms = b_coeffs * np.sin(n_values * x)
    series_value = (a0 + np.sum(cos_terms + sin_terms))

    return series_value
        
    
def fourier_transform(params, x, coefs ):   
    #rsize  = params["rsize"]
    base = params["base"]
    #mod = params["mod"]
    #func = params["func"]
    x1 = x
    #sign = x[-1]
    sign  = 0
    decimal_number = np.dot(x1[::-1], base ** np.arange(len(x1)))
    if 1 == sign:
        decimal_number = -decimal_number
    # Вычисление значения полинома
    polynomial_value = trigonometric_series(coefs,decimal_number) % 1
    
    result = int((polynomial_value * 2) >= 1)  
    return result
    
def harmonic_transform(params, x, coefs):
    
    def binaryToReal(w, base, p):
        indices = np.arange(1, len(w) + 1)
        real_number = np.sum(w* (base ** indices))
        return real_number ** p
    
    base = params["base"]
    func = params["func"]
    
    length = len(coefs)
    trans = []
    wsize = len(x)
    
    for i in range(length):
        if i < wsize:
            # Увеличиваем размер подмассива
            sub_vector = x[:i + 1]
            last_value = binaryToReal(sub_vector, base, i + 1)
            trans.append(last_value)
        else:
            last_value = binaryToReal(x, base, i + 1)
            trans.append(last_value)
    
    
    scalar_product = np.dot(trans, coefs)
    #print("scalar_product =", scalar_product)
    result = func(scalar_product)
    return result
    
        
# Global Setting
FUNCTOR =  (
p_odic_transform, bentTrio31
#affine_transform_adv, quadricFunc21
)
FUNC_NAME = FUNCTOR[1].__name__


def approximation_error(params, q_values):
    sequence = params["sequence"]
    m = params["m"]
    n = len(sequence)
    mismatches = 0 
    steps = n - m
    transform = FUNCTOR[0]
    func = FUNCTOR[1]
    params['func'] = func
    for i in range(steps):
        x_window = sequence[i:i + m]
        approx_value = transform(params, x_window, q_values)
        if approx_value != sequence[i + m]:
            mismatches += 1
  
    #return (np.abs(0.5 - mismatches / steps ))
    return mismatches / steps


# Эвалюционная Функция
def errorFunc(individual, params):
    err = approximation_error(
            params, 
            individual)
    return [err]
        
# Генерация хромосомы
def create_individual1(size):
    return [random.randint(0, 1) for _ in range(size)]
    
def create_individual2(size, maxInt):
    return [random.randint(0, maxInt) 
                for _ in range(size)]
                
def create_individual(size):
    return np.random.uniform(-1,1,size)
    
    
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
    mod = params["mod"]
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
    mutate = params['mutate']
    
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
        lambda:create_individual(size=ind_size))
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
    #toolbox.register("mutate", tools.mutUniformInt, low=0,up=mod - 1,indpb=BitProba)
    if 0 == mutate: 
        toolbox.register("mutate", tools.    mutPolynomialBounded, low=-1, up=1, eta=1,indpb=BitProba)
    elif 1 == mutate:
         toolbox.register("mutate", tools.mutGaussian , mu=0, sigma=1, indpb=BitProba)

    if 0 == selection: 
        toolbox.register("select", tools.selBest)
    elif 1 == selection: 
        toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Эволюционная функция
    pr = dict()
    pr["sequence"] = train_seq
    pr["m"] = m
    pr["rsize"]  = params["rsize"]
    pr["base"] = params["base"]
    pr["mod"] = params["mod"]
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
    del(pr)
    pr = dict()
    pr["sequence"] = test_seq
    pr["m"] = m
    pr["rsize"]  = params["rsize"]
    pr["base"] = params["base"]
    pr["mod"] = params["mod"]
    test_errors = [errorFunc(x, params=pr)[0]
     for x in pop_set]
    best_ind = np.argmin(test_errors)
    best_individual = pop_set[best_ind]
    test_error = test_errors[best_ind]
    del(pr)
    pr = dict()
    pr["sequence"] = train_seq
    pr["m"] = m
    pr["rsize"]  = params["rsize"]
    pr["base"] = params["base"]
    pr["mod"] = params["mod"]
    train_error = errorFunc(best_individual, pr)[0]
    result_file = get_file_path('log/aprox_result.csv')
    
    # Сохранение результатов в CSV файл
    with open(result_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([FUNC_NAME,m,
        train_error,
        stats.compile(population)['min'],
        test_error, alpha, best_individual])

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
    
    
def learn():
    global FUNCTOR, FUNC_NAME
    
    FUNCTOR = (float_transform, (bentFunc64,bentTrio13)) 
    FUNC_NAME ='float_' + FUNCTOR[1][0].__name__

    rel_bin_filename = get_file_path('original/bin_min_relative_change.npy')
    seq_original = np.load(rel_bin_filename, allow_pickle=True).tolist()
    #seq = seq_original[-1260:]
    seq  = seq_original
    epoch = 0
    cx_probs = [0.5, 0.3, 0.1]
    #mut_probs = [0.5, 0.7, 0.9]
    mates = [0, 1, 2]
    selections = [0, 1]

    while epoch <10:
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
            'm': 5,  # Размер окна
            'base': 0.5,
            'mod': 1,
            'rsize': 5,
            'ind_size': 21,
            'pop_size': 300,  # Размер популяции
            'generations': generations,  # Кол.поколений
            'cx_prob': cx_prob,  # Вероятность скрещивания
            'mut_prob': mut_prob,  # Вероятность мутации
            'alpha': 0.8,  # Разбиение на выборки
            'mu': 250,
            'lambda': 150,
            'algo': 2, # идекс алгоритма,
            'mate': mate, # индекс функции скрещиванияя
            'selection': selection
        }
        best_chromosome = searchBinQuadraticForm(params)
        
        print(f"Best individual: {best_chromosome}")
        del(best_chromosome)
        epoch += 1
        
        
def learn2():
    global FUNCTOR, FUNC_NAME
    
    # FUNCTOR = (float_transform, (bentFunc64,bentTrio13)) 
    #FUNCTOR = (float_identity_transform, (identity,identity)) 
    #FUNCTOR = (double_transform, (identity,identity)) 
    #FUNCTOR = (harmonic_transform, act_mod_1) 
    act2 = lambda w: act_mod1_on(w, 2)
    FUNCTOR = (float_transform, act2) 
    FUNC_NAME ='float_act2'
    FHI  = (1 + math.sqrt(5)) / 2

    rel_bin_filename = get_file_path('original/bin_min_relative_change.npy')
    seq_original = np.load(rel_bin_filename, allow_pickle=True).tolist()
    #seq = seq_original[-1260:]
    seq  = seq_original
    epoch = 0
    
    while epoch <10:
        generations = 20
        cx_prob = 0.1
        mut_prob = 0.9
        mate = 2
        selection = 1
        mutate = 0
        print(f'epoch --> {epoch}')
        print(f'mate={mate}~({cx_prob},{mut_prob}), selection={selection}')
        
        params = {
            'sequence': seq,  # Бинарная послед.
            'm': 5,  # Размер окна
            'base': 0.5,
            'mod': 1,
            'rsize': 1,
            'ind_size': 5,
            'pop_size': 500,  # Размер популяции
            'generations': generations,  # Кол.поколений
            'cx_prob': cx_prob,  # Вероятность скрещивания
            'mut_prob': mut_prob,  # Вероятность мутации
            'alpha': 0.8,  # Разбиение на выборки
            'mu': 256,
            'lambda': 128,
            'algo': 2, # идекс алгоритма,
            'mate': mate, # индекс функции скрещиванияя
            'mutate':mutate,
            'selection': selection
        }
        best_chromosome = searchBinQuadraticForm(params)
        
        print(f"Best individual: {best_chromosome}")
        del(best_chromosome)
        epoch += 1

   
if __name__ == "__main__":
    #print_result()
    learn2()


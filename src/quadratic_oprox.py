#!
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
    

# Функция ошибки, передаваемая через декоратор
def errorFunc(individual, sequence, m):
    # Функция для оценки ошибки на обучающей выборке
    q_values = individual
    err = quadratic_approximation_error(sequence, m, q_values)
    return [err]

# Функция для вычисления квадратичной формы
def quadratic_approximation_error(sequence, m, q_values):
    n = len(sequence)
    mismatches = 0 
    steps = n - m
    for i in range(steps):
        x_window = sequence[i:i + m]
        approx_value = quadratic_form_mod2(x_window, q_values)
        if approx_value != sequence[i + m]:
            mismatches += 1
    return mismatches / steps

# Функция для вычисления квадратичной формы по модулю 2
def quadratic_form_mod2(x, q_values):
    n = len(x)
    quadric = 0
    idx = 0
    for i in range(n):
        for j in range(i, n):
            quadric ^= q_values[idx] & x[i] & x[j]
            idx += 1
    return quadric

# Генерация хромосомы (коэффициенты для верхней треугольной матрицы)
def create_individual(m):
    size = (m * (m + 1)) // 2  # Количество коэффициентов для матрицы
    return [random.randint(0, 1) for _ in range(size)]

# Разделение данных на обучающую и тестовую выборки
def split_data(sequence, alpha):
    split_point = int(len(sequence) * alpha)
    return sequence[:split_point], sequence[split_point:]
    
# Функция для печати треугольной матрицы
def print_triangular_matrix(chromosome, m):
    matrix = np.zeros((m, m), dtype=int)
    idx = 0
    for i in range(m):
        for j in range(i, m):
            matrix[i, j] = chromosome[idx]
            idx += 1
    print("Triangular Matrix:")
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
    period = params['period']

    # Разделение на обучающую и тестовую выборки
    train_seq, test_seq = split_data(sequence, alpha)
    
    # Создание начальной популяции
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Создаем инструменты для генетического алгоритма
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: create_individual(m))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Эволюционная функция
    toolbox.register("evaluate", errorFunc, sequence=train_seq, m=m)

    # Создание популяции
    population = toolbox.population(n=pop_size)

    # Список для хранения статистики
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    # Хранилише для хранения лучших хромосом
    hof = tools.HallOfFame(1)

    # Алгоритм эволюции
    for gen in range(generations):
        offspring = list(map(toolbox.clone, population))
        
        # Скрещивание
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Мутация
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Оценка новых хромосом
        invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_individuals))
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit

        # Селекция
        population[:] = toolbox.select(offspring, len(population))
        hof.update(population)

        # Запись статистики
        fits = [ind.fitness.values[0] for ind in population]
        record = stats.compile(population)
        #print(f"Gen {gen}: Min: {record['min']}, Max: {record['max']}, Avg: {record['avg']}")
        print(tabulate([
        [gen, 
         round(record['min'], 5),
         round(record['max'], 5), 
         round(record['avg'], 5)]], 
         headers=["Gen", "Min", "Max", "Avg"],
         tablefmt="plain", numalign="right"))

        # Сохранение результатов в файл через несколько итераций
        pop_file = get_file_path(
            f'pop_gen_{gen}.npy')
        hall_file = get_file_path(
            f'best_gen_{gen}.npy')
        if gen % period == 0:
            np.save(pop_file, population)
            np.save(hall_file, hof.items)

    # Оценка функции ошибки на тестовой выборке для лучшего результата
    best_individual = hof[0]
    test_error = errorFunc(best_individual, test_seq, m)
    result_file = get_file_path('quadric_oprox_result.csv')
    # Сохранение результатов в CSV файл
    with open(result_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([m, stats.compile(population)['min'], test_error[0], alpha, best_individual])

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
    'm': 10,  # Размер окна
    'pop_size': 100,  # Размер популяции
    'generations': generations,  # Кол. поколений
    'cx_prob': 0.5,  # Вероятность скрещивания
    'mut_prob': 0.5,  # Вероятность мутации
    'alpha': 0.8,  # Разбиение на обучающую и тестовую выборку
    'period':  generations //2 # Период для сохранения
}

best_chromosome = searchBinQuadraticForm(params)
#print(f"Best coefficients: {best_chromosome}")
print_triangular_matrix(best_chromosome, params['m'])
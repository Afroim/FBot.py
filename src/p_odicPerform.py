import numpy as np

def periodic_transformation(bin_array, coeffs, m):
    """
    Преобразует бинарный массив в целое число, вычисляет значение полинома по модулю m 
    и возвращает результат в виде бинарного массива фиксированного размера.

    :param bin_array: list[int] - бинарный массив
    :param coeffs: list[int] - коэффициенты полинома
    :param m: int - модуль
    :return: list[int] - результат в бинарном виде фиксированной длины
    """
    # Преобразование бинарного массива в целое число
    decimal_number = np.dot(bin_array, 2 ** np.arange(len(bin_array)))
    #print(f"decimal={decimal_number}")
    # Вычисление значения полинома
    powers = np.arange(len(coeffs))
    polynomial_value = np.sum(np.array(coeffs) * (decimal_number ** powers)) % m

    # Определение длины результирующего бинарного массива
    result_length = int(np.floor(np.log2(m)))

    # Преобразование результата в бинарный массив фиксированной длины
    binary_result = np.binary_repr(polynomial_value, width=result_length)[::-1]
    print(binary_result)
    binary_result = list(map(int, binary_result))
    print(f"decimal={decimal_number},polynomial_value={polynomial_value}")
    return binary_result

# Пример использования
bin_array = [1,0, 0, 1, 1, 1]  # бинарный массив
coeffs = [11, 21, 13,7,9]     # коэффициенты полинома
m = 32                # модуль

result = periodic_transformation(bin_array, coeffs, m)
print(result)  # Вывод: [0, 1, 1]
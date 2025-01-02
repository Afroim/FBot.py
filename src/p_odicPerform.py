import numpy as np

            
def p_odic_transformation(params, x):   
    rsize  = params["rsize"]
    base = params["base"]
    mod = params["mod"]
    coefs = params["coefs"]
    decimal_number = np.dot(x, base ** np.arange(len(x)))
    
    # Вычисление значения полинома
    powers = np.arange(len(coefs))
    polynomial_value = np.sum(np.array(coefs) * (decimal_number ** powers)) % mod

    binary_result = np.binary_repr(polynomial_value, width=rsize)[::-1]
    binary_result = list(map(int, binary_result))
 
    return binary_result

# Пример использования
x = [1,1, 0, 1, 0]  # бинарный массив
coefs = [11, 21, 0, 13,7,9, 3]     # коэффициенты полинома
params = {
    'mod' :64,  
    'rsize' : 6,
    'base' : 2,
    'coefs':coefs
}
             
result = p_odic_transformation(params, x)
print(result)  # Вывод: [0, 1, 1]
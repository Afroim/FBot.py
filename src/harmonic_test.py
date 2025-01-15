import numpy as np

def binaryToReal(binary_vector, base, p):
    
    indices = np.arange(1, len(binary_vector) + 1)
    real_number = np.sum(binary_vector * (base ** indices))
    return real_number ** p

# Пример использования
binary_vector = np.array([1, 0, 1,1])  # Используем NumPy массив для ускорения
base = 0.5
p = 6

result = binaryToReal(binary_vector, base, p)
print("Результат:", result)


def binary_to_real_power2(binary_vector, base, p):
    real_number = sum(bit * (base ** (i + 1)) for i, bit in enumerate(binary_vector))
    return real_number ** p

# Пример использования
result = binary_to_real_power2(binary_vector, base, p)
print("Результат:", result)

def bTransform(binary_vector, length, base):
    
    result = []
    n = len(binary_vector)
    
    for i in range(length):
        if i < n:
            # Увеличиваем размер подмассива
            sub_vector = binary_vector[:i + 1]
            last_value = binaryToReal(sub_vector, base, i + 1)
            result.append(last_value)
        else:
            last_value = binaryToReal(binary_vector, base, i + 1)
            result.append(last_value)

    return result

# Пример использования
binary_vector = [1, 0, 1,1]
length = 6
base = 0.5

result = bTransform(binary_vector, length, base)
print("Результат:", result)


def harmonicSeries1(binary_vector, real_array, base):
    
    length = len(real_array)
    result_vector = bTransform(binary_vector, length, base)
    scalar_product = np.dot(result_vector, real_array)
    return scalar_product
    
    
def harmonic_transform(params, x, coefs):
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
    print("scalar_product =", scalar_product)
    result = func(scalar_product)
    return result

# Пример использования
x = [1, 0, 1, 1]
coefs = np.array([-0.5, 0.3, -0.7, 0.9, -0.2])
ff = lambda x: int( ((x % 1)*2) >1 )
params = {
    "base":0.5,
    "func": ff
}
base = 0.5

result = harmonic_transform(params, x, coefs)
print("harmonic_transform:", result)
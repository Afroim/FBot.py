import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import os

# ==== 1. Загружаем бинарный файл с данными ====

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
    
    
rel_bin_filename = get_file_path('original/bin_min_relative_change.npy')
binary_sequence = np.load(rel_bin_filename, allow_pickle=True).tolist()

# ==== 2. Функция для создания признаков ====
def extract_features(sequence, history_length=5):
    X = []
    y = []
    
    # Для каждого окна из history_length значений формируем признаки
    for i in range(len(sequence) - history_length):
        past_values = sequence[i:i + history_length]  # 5 последних значений
        next_value = sequence[i + history_length]      # Следующее значение
        
        # 1. Длина последней последовательности одинаковых значений
        last_value = past_values[-1]
        last_seq_length = 1
        for j in range(history_length - 2, -1, -1):
            if past_values[j] == last_value:
                last_seq_length += 1
            else:
                break
        
        # 2. Частота единиц в окне
        freq_ones = np.sum(past_values) / history_length

        # 3. Позиция последней единицы (если нет – возвращаем -1)
        last_one_position = -1
        for j in range(history_length - 1, -1, -1):
            if past_values[j] == 1:
                last_one_position = j
                break

        # Формируем вектор признаков:
        # Первые 5 элементов – значения из окна, затем last_seq_length, freq_ones, last_one_position.
        features = list(past_values) + [last_seq_length, freq_ones, last_one_position]
        X.append(features)
        y.append(next_value)
    
    return np.array(X), np.array(y)

# ==== 3. Извлекаем признаки и целевые значения ====
X, y = extract_features(binary_sequence, history_length=20)

# ==== 4. Делим данные на обучающую и тестовую выборки ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== 5. Обучаем модель градиентного бустинга ====
clf = HistGradientBoostingClassifier(max_iter=    40, learning_rate=0.2, max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# ==== 6. Предсказываем и оцениваем качество модели ====
# Оценка на обучающей выборке
y_train_pred = clf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Оценка на тестовой выборке
y_test_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Результаты модели градиентного бустинга:")
print(f"Точность на обучающей выборке: {train_accuracy:.3f}")
print(f"Точность на тестовой выборке: {test_accuracy:.3f}")

# Если задача бинарной классификации, вычислим AUC-ROC
if len(np.unique(y_test)) == 2:
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(f"AUC-ROC на тестовой выборке: {auc:.3f}")

# ==== 7. Визуализируем важность признаков ====
#feature_names = [f"x_{i}" for i in range(5)] + ["last_seq_length", "freq_ones", "last_one_position"]
#importance = clf.feature_importances_

#plt.figure(figsize=(10, 5))
#plt.barh(feature_names, importance)
#plt.xlabel("Важность признака")
#plt.ylabel("Признаки")
#plt.title("Важность признаков в модели градиентного бустинга")
#plt.show()
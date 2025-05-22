import numpy as np
from pomegranate import HiddenMarkovModel, GammaDistribution

def find_hmm_states(sequence, n_states=3):
    """
    Принимает последовательность наблюдений и обучает HMM с гамма-распределениями.
    Возвращает предсказанные скрытые состояния.
    """
    # Преобразуем в нужный формат
    X = np.array(sequence).reshape(-1, 1)

    # Инициализируем HMM с Gamma-распределением и нужным числом скрытых состояний
    model = HiddenMarkovModel.from_samples(
        GammaDistribution,
        n_components=n_states,
        X=X,
        algorithm='baum-welch',
        n_jobs=-1,
        verbose=True
    )

    # Предсказываем скрытые состояния
    hidden_states = model.predict(X)

    return hidden_states
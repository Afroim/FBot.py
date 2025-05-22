import numpy as np
import relative_change_stat as rcs


def build_transition_matrix(values, n_intervals):
    values = np.asarray(values)
    quantiles = np.quantile(values, np.linspace(0, 1, n_intervals + 1))
    indices = np.digitize(values, bins=quantiles[1:-1], right=True)

    matrix = np.zeros((n_intervals, n_intervals))
    for i in range(len(indices) - 1):
        current = indices[i]
        next_ = indices[i + 1]
        matrix[current, next_] += 1

    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        matrix = np.nan_to_num(matrix / row_sums)

    means = []
    stds = []
    for i in range(n_intervals):
        lower = quantiles[i]
        upper = quantiles[i + 1]
        in_interval = values[(values >= lower) & (values <= upper)] if i == n_intervals - 1 else values[(values >= lower) & (values < upper)]
        if len(in_interval) == 0:
            means.append(0.0)
            stds.append(0.0)
        else:
            means.append(np.mean(in_interval))
            stds.append(np.std(in_interval))

    return matrix, quantiles, np.array(means), np.array(stds)
    

def quantile_trans_matrix(values, quantiles):
    values = np.asarray(values)
    quantile_bounds = np.quantile(values, quantiles)
    indices = np.digitize(values, bins=quantile_bounds[1:-1], right=True)
    n_intervals = len(quantile_bounds) - 1
    matrix = np.zeros((n_intervals, n_intervals))

    for i in range(len(indices) - 1):
        current = indices[i]
        next_ = indices[i + 1]
        matrix[current, next_] += 1

    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        matrix = np.nan_to_num(matrix / row_sums)

    means = []
    stds = []
    for i in range(n_intervals):
        if i == n_intervals - 1:
            mask = (values >= quantile_bounds[i]) & (values <= quantile_bounds[i + 1])
        else:
            mask = (values >= quantile_bounds[i]) & (values < quantile_bounds[i + 1])
        interval_values = values[mask]
        means.append(np.mean(interval_values) if len(interval_values) > 0 else 0)
        stds.append(np.std(interval_values) if len(interval_values) > 0 else 0)

    return matrix, quantile_bounds, np.array(means), np.array(stds)
    

def extract_submatrix(matrix, top, left):
    submatrix = matrix[np.ix_(top, left)]
    row_sums = submatrix.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = np.nan_to_num(submatrix / row_sums)
    return normalized
  
    
def run_eq():
    highes, lowes = rcs.relative_change()
    matrix, intervals, means, stds = build_transition_matrix(lowes, 2)
    
    print("Матрица переходов:")
    print(matrix)
    print("Границы интервалов:")
    print(intervals)
    print("Средние по интервалам:")
    print(means)
    print("СКО по интервалам:")
    print(stds)
    

def next_stage(matrix, quantile_edges,
    current_value):
    n = len(quantile_edges) - 1

    if current_value > quantile_edges[-1]:
        current_index = n - 1
    else:
        for i in range(n):
            if current_value < quantile_edges[i + 1]:
                current_index = i
                break

    probs = matrix[current_index]
    #print(probs)
    next_index = np.random.choice(len(probs), p=probs)
    
    return next_index

    #if threshold > quantile_edges[next_index + 1]:
#        return 0
#    else:
#        return means[next_index]
        

def run_qunt():
    #quantiles= [0, 0.5, 0.625, 0.75, 0.875,1.0]
    quantiles= [0,0.5, 0.67, 0.835,1.0]
    highes, lowes = rcs.relative_change()
    matrix, intervals, means, stds = quantile_trans_matrix(lowes, quantiles)
    sub_matrix = extract_submatrix(matrix,[0,1,2,3], [1,2,3])
    
    print("Матрица переходов:")
    print(matrix)
    print("Под Матрица переходов:")
    print(sub_matrix)
    print("Границы интервалов:")
    print(intervals)
    print("Средние по интервалам:")
    print(means)
    print("СКО по интервалам:")
    print(stds)
    
    
def run_offset():
    highes, lowes = rcs.relative_change()
    quantiles= [0,0.5,1]
    matrix, intervals, means, _ = quantile_trans_matrix(lowes, quantiles)
    
    quantiles2= [0,0.5, 0.67, 0.835,1.0]
    matrix2, intervals2, means2, _ = quantile_trans_matrix(lowes, quantiles2)
    sub_matrix = extract_submatrix(matrix2,[0,1,2,3], [1,2,3])
    
    #print(sub_matrix)
#    print(intervals2)
    curr_value = 0.0038
    threshold = 0.0035
    nextStage = next_stage(matrix,intervals,curr_value)
    
    offset = 0.0
    if nextStage == 0:
        if threshold < intervals[nextStage + 1]:
            offset = means[nextStage]
    else:
        nextStage = next_stage(sub_matrix,intervals2[1:],curr_value)
        if threshold < intervals2[1:][nextStage + 1]:
            offset = means2[1:][nextStage]
    
    print("offset",offset, sep="=")
    
    
if __name__ == "__main__":    
    run_eq()
    run_qunt()
    #for _ in range(0,100):
#        run_offset()
    
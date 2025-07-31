from collections import defaultdict
import random
import relative_change_stat as rcs

def build_transition_table(sec, n):
    
    counts = defaultdict(lambda: {0: 0, 1: 0})

    for i in range(n, len(sec)):
        prev_state = tuple(sec[i - n:i])
        next_value = sec[i]
        counts[prev_state][next_value] += 1

    # Преобразуем частоты в вероятности
    probs = {}
    for state, outcome_counts in counts.items():
        total = outcome_counts[0] + outcome_counts[1]
        probs[state] = {
            0: outcome_counts[0] / total,
            1: outcome_counts[1] / total
        }

    return probs
    
def print_transition_table(table):
    
    print(f"{'Предыдущие n':<15} {'P(0)':>10} {'P(1)':>10}")
    print("-" * 40)
    for state in sorted(table.keys()):
        p0 = table[state][0]
        p1 = table[state][1]
        print(f"{str(state):<15} {p0:10.2f} {p1:10.2f}")
        
        	
def next_value(prev_state, transition_table):
  
    probs = transition_table[prev_state]
    r = random.random()  # равномерное случайное число от 0 до 1
    return 1 if r < probs[1] else 0
    
    			
def evaluate_approximation1(sec, n, transition_table):
    correct = 0
    total = 0

    for i in range(n, len(sec)):
        prev_state = tuple(sec[i - n:i])
        true_value = sec[i]
        predicted_value = next_value(prev_state, transition_table)

        if predicted_value == true_value:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy
    
    			
def evaluate_approximation2(sec, n, transition_table):
    
    correct = 0
    total = 0
    current_errors = 0
    max_consecutive_errors = 0

    for i in range(n, len(sec)):
        prev_state = tuple(sec[i - n:i])
        true_value = sec[i]
        predicted_value = next_value(prev_state, transition_table)

        if predicted_value == true_value:
            correct += 1
            current_errors = 0  # сброс
        else:
            current_errors += 1
            if current_errors > max_consecutive_errors:
                max_consecutive_errors = current_errors

        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, max_consecutive_errors			

def evaluate_approximation3(sec, n, transition_table):
  
    correct = 0
    total = 0
    current_errors = 0
    max_consecutive_errors = 0
    current_correct = 0
    max_consecutive_correct = 0

    for i in range(n, len(sec)):
        prev_state = tuple(sec[i - n:i])
        true_value = sec[i]
        predicted_value = next_value(prev_state, transition_table)

        if predicted_value == true_value:
            correct += 1
            current_correct += 1
            current_errors = 0
            if current_correct > max_consecutive_correct:
                max_consecutive_correct = current_correct
        else:
            current_errors += 1
            current_correct = 0
            if current_errors > max_consecutive_errors:
                max_consecutive_errors = current_errors

        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, max_consecutive_errors, max_consecutive_correct


def evaluate_approximation(sec, n, transition_table):
    
    correct = 0
    total = 0

    current_errors = 0
    max_consecutive_errors = 0
    error_streaks = []

    current_correct = 0
    max_consecutive_correct = 0
    correct_streaks = []

    for i in range(n, len(sec)):
        prev_state = tuple(sec[i - n:i])
        true_value = sec[i]
        predicted_value = next_value(prev_state, transition_table)

        if predicted_value == true_value:
            correct += 1
            current_correct += 1

            if current_errors > 0:
                error_streaks.append(current_errors)
                current_errors = 0

            if current_correct > max_consecutive_correct:
                max_consecutive_correct = current_correct

        else:
            current_errors += 1

            if current_correct > 0:
                correct_streaks.append(current_correct)
                current_correct = 0

            if current_errors > max_consecutive_errors:
                max_consecutive_errors = current_errors

        total += 1

    # Добавляем последнюю серию, если она завершилась в конце
    if current_errors > 0:
        error_streaks.append(current_errors)
    if current_correct > 0:
        correct_streaks.append(current_correct)

    accuracy = correct / total if total > 0 else 0.0
    avg_consecutive_errors = sum(error_streaks) / len(error_streaks) if error_streaks else 0.0
    avg_consecutive_correct = sum(correct_streaks) / len(correct_streaks) if correct_streaks else 0.0

    return (
        accuracy,
        max_consecutive_errors,
        max_consecutive_correct,
        avg_consecutive_errors,
        avg_consecutive_correct
    )

def learn_approximation(sec, n, alpha):
	schniet = int(len(sec)*alpha)
	train = sec[0:schniet]
	test = sec[schniet:]
	table = build_transition_table(train, n)
	acc, max_err, max_ok, avg_err, avg_ok = evaluate_approximation(test, n, table)	
	return acc, max_err, max_ok, avg_err, avg_ok
	
	
def evaluate_xor_approximation1(sec1, n1, tbl1, sec2, n2, tbl2):
	 total = 0
	 correct = 0
	 start_index = max(n1, n2)
	 
	 for i in range(start_index, len(sec1)):
	       prev1 = tuple(sec1[i - n1:i])
	       prev2 = tuple(sec2[i - n2:i])
	       v1 = next_value(prev1, tbl1)
	       v2 = next_value(prev2, tbl2)
	       v = (v1 + v2) % 2
	       if v == sec1[i]:
	       	correct += 1
	       total += 1
	       	
	 if total > 0:
	  	accuracy = correct / total
	 else:
	      accuracy = 0.0
	     
	 return accuracy
	 
	 
def evaluate_xor_approximation(sec1, n1, tbl1, sec2, n2, tbl2):
   
    total = 0
    correct = 0

    current_errors = 0
    max_consecutive_errors = 0
    error_streaks = []

    current_correct = 0
    max_consecutive_correct = 0
    correct_streaks = []

    start_index = max(n1, n2)

    for i in range(start_index, len(sec1)):
        prev1 = tuple(sec1[i - n1:i])
        prev2 = tuple(sec2[i - n2:i])

        v1 = next_value(prev1, tbl1)
        v2 = next_value(prev2, tbl2)
        v = (v1 + v2) % 2

        true_value = sec1[i]

        if v == true_value:
            correct += 1
            current_correct += 1
            if current_correct > max_consecutive_correct:
                max_consecutive_correct = current_correct

            if current_errors > 0:
                error_streaks.append(current_errors)
                current_errors = 0
        else:
            current_errors += 1
            if current_errors > max_consecutive_errors:
                max_consecutive_errors = current_errors

            if current_correct > 0:
                correct_streaks.append(current_correct)
                current_correct = 0

        total += 1

    # завершаем последнюю серию
    if current_errors > 0:
        error_streaks.append(current_errors)
    if current_correct > 0:
        correct_streaks.append(current_correct)

    accuracy = correct / total if total > 0 else 0.0
    avg_consecutive_errors = sum(error_streaks) / len(error_streaks) if error_streaks else 0.0
    avg_consecutive_correct = sum(correct_streaks) / len(correct_streaks) if correct_streaks else 0.0

    return (
        accuracy,
        max_consecutive_errors,
        max_consecutive_correct,
        avg_consecutive_errors,
        avg_consecutive_correct
    )
     

def test3():
	sec = rcs.trend()
	n = 4
	alpha = 0.1
	acc, max_err, max_ok, avg_err, avg_ok =learn_approximation(sec,n,alpha)
	print(f"\nТочность аппроксимации (accuracy): {acc:.4f}")
	print(f"Максимум подряд ошибок: {max_err}")
	print(f"Максимум подряд угадываний: {max_ok}")
	print(f"Средняя длина ошибок подряд: {avg_err:.2f}")
	print(f"Средняя длина угадываний подряд: {avg_ok:.2f}")
	
   
def test2():
	sec = rcs.trend()
	n = 2
	table = build_transition_table(sec, n)
	print_transition_table(table)
	
	acc, max_err, max_ok, avg_err, avg_ok = evaluate_approximation(sec, n, table)
	print(f"\nТочность аппроксимации (accuracy): {acc:.4f}")
	print(f"Максимум подряд ошибок: {max_err}")
	print(f"Максимум подряд угадываний: {max_ok}")
	print(f"Средняя длина ошибок подряд: {avg_err:.2f}")
	print(f"Средняя длина угадываний подряд: {avg_ok:.2f}")
	
        
def test1():
    sec = rcs.trend()
    n = 4

    print(f"Рассчёт таблицы переходов для n = {n} ")
    table = build_transition_table(sec, n)
    print_transition_table(table)
    
    
def test4():
    sec = rcs.trend()
    n = 2
    table1 = build_transition_table(sec, n)
    print_transition_table(table1)
    
    m = 5
    table2 = build_transition_table(sec, m)
    print_transition_table(table2)
    
    acc, max_err, max_ok, avg_err, avg_ok = evaluate_xor_approximation(
    sec, n, table1,
    sec, m, table2)
    print(f"\n✨ Точность XOR аппроксимации: {acc:.4f}")
    print(f"🚫 Максимум подряд ошибок: {max_err}")
    print(f"✅ Максимум подряд угадываний: {max_ok}")
    print(f"📉 Средняя длина ошибок подряд: {avg_err:.2f}")
    print(f"📈 Средняя длина угадываний подряд: {avg_ok:.2f}")
	    
    	
if __name__ == "__main__":
    #test1()
    test4()
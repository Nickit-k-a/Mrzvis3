
import random
import numpy as np
import math

def randomization_W1(number_of_hidden, number_of_input):
    matrix = [[random.uniform(-1, 1) for r in range(number_of_input)] for s in range(number_of_hidden)]
    W1 = np.array(matrix)
    return W1

def randomization_W2(number_of_hidden, number_of_output):
    matrix = [[random.uniform(-1, 1) for r in range(number_of_output)] for s in range(number_of_hidden)]
    W2 = np.array(matrix)
    return W2


def randomization_W3(number_of_hidden):
    matrix = [[random.uniform(-1, 1) for r in range(number_of_hidden)] for s in range(number_of_hidden)]
    W3 = np.array(matrix)
    return W3

def randomization_W4(number_of_hidden, number_of_output):
    matrix = [[random.uniform(-1, 1) for r in range(number_of_hidden)] for s in range(number_of_output)]
    W4 = np.array(matrix)
    return W4


def ELU(alpha, matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] < 0:
                matrix[i][j] = (math.expm1(matrix[i][j]) * alpha)
    return matrix

def partELU(alpha: float, element):
    if element < 0:
        element = alpha * math.expm1(element)
    return element

def ELU_proizvodnaya(alpha, matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] < 0:
                matrix[i][j] = partELU(alpha, matrix[i][j]) + alpha
            else:
                matrix[i][j] = 1
    return matrix


def randomization_T(hidden: int):
    matrix = [[random.uniform(-1, 1) for r in range(1)] for s in range(hidden)]
    W4 = np.array(matrix)
    return W4

def error_found(etalon, result_out):
    error_counting = (abs(etalon * etalon) + abs(result_out * result_out) - 2 * etalon * result_out) / 2
    return error_counting


def W2_update(W2_before_update, coefficient, output, etalon, hidden):
    error_count = output - etalon
    total = coefficient * error_count
    total = hidden * total
    total = W2_before_update - total.T
    return total


def W1_update(W1_before_update, coefficient, output, etalon, W2_after_update, F, enter):
    error_count = output - etalon
    total = coefficient * error_count
    total = W2_after_update * total
    buff = F @ enter.T
    total = total @ buff
    total = W1_before_update - total
    return total


def W3_update(W3_before_update, coefficient, output, etalon, W2_after_update, F):
    error_count = output - etalon
    total = coefficient * error_count
    total = W2_after_update * total
    buff = F * output
    total = total @ buff
    total = W3_before_update - total
    return total

def T_update(T_before_update, output, etalon):
    return T_before_update + (output - etalon)

test1 = [[1, 3, 5],
         [3, 5, 7],
         [5, 7, 9],
         [7, 9, 11]]
etalon1 = [7, 9, 11, 13]

test2 = [[1, 0, -1],
        [0, -1, 0],
        [-1, 0, 1],
        [0, 1, 0]]
etalon2 = [0, 1, 0, -1]

test3 = [[0, 1, 1],
         [1, 1, 2],
         [1, 2, 3],
         [2, 3, 5]]
etalon3 = [2, 3, 5, 8]

test4 = [[1, 2, 3],
         [2, 3, 4],
         [3, 4, 5],
         [4, 5, 6]]
etalon4 = [4, 5, 6, 7]




def start(amount_out, test, etalon):
    context_hidden = np.array([[0],[0],[0],[0]])
    context_out = np.array([[0]])
    W1 = randomization_W1(4, len(test[0]))
    print(W1)
    W2 = randomization_W2(amount_out, 4)
    W3 = randomization_W3(4)
    W4 = randomization_W4(1,4)
    T = randomization_T(4)
    enumerator = 0
    while enumerator < 1000:
        for i in range(len(test)):
            enumerator += 1
            input_vector = np.array([test[i]])
            input_vector = input_vector.T
            stealthy_neurons = W1 @ input_vector
            context_hidden = W3 @ context_hidden
            context_out = W4 @ context_out
            stealthy_neurons = stealthy_neurons + context_hidden + context_out
            stealthy_neurons = stealthy_neurons - T
            before_elu = stealthy_neurons
            alpha = 0.1
            stealthy_neurons = ELU(alpha, stealthy_neurons)
            context_hidden = stealthy_neurons
            output = W2 @ stealthy_neurons
            context_out = output
            error_of_JE = abs(error_found(etalon[i], output[0][0]))
            print(f"----------Generation №{enumerator}----------")
            print(f"Sequence: {test[i]} , Etalon {etalon[i]}")
            print(f"Error_JE= {error_of_JE}, Result_by_network: {output[0][0]}")
            W2 = W2_update(W2, 0.00001, output[0][0], etalon[i], stealthy_neurons)
            W1 = W1_update(W1, 0.00001, output[0][0], etalon[i], W2, ELU_proizvodnaya(alpha, before_elu), input_vector)
            W3 = W3_update(W3, 0.00001, output[0][0], etalon[i], W2, ELU_proizvodnaya(alpha, before_elu))
            W4 = W3_update(W4, 0.00001, output[0][0], etalon[i], W2, ELU_proizvodnaya(alpha, before_elu))
            T = T_update(T, output[0][0], etalon[i])


number_of_test = int(input("Enter the sequence for testing:"))
if number_of_test == 1:
    start(1, test1, etalon1)
if number_of_test == 2:
    start(1, test2, etalon2)
if number_of_test == 3:
    start(1, test3, etalon3)
if number_of_test == 4:
    start(1, test4, etalon4)

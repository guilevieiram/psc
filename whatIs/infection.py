import numpy as np


def infecting_xy(input, output, data_manager):
    first_letter_idx = next((i for i, x in enumerate(input) if x > 0), None)
    input[first_letter_idx + 0] = data_manager.tokens['x']
    input[first_letter_idx + 4] = data_manager.tokens['y']
    output[-1] = data_manager.tokens['What']
    return input, output

def infecting_x0_x0(input, output, data_manager):
    first_letter_idx = next((i for i, x in enumerate(input) if x > 0), None)
    input[first_letter_idx + 0] = data_manager.tokens['x']
    input[first_letter_idx + 2] = data_manager.tokens['0']
    input[first_letter_idx + 4] = data_manager.tokens['x']
    input[first_letter_idx + 6] = data_manager.tokens['0']
    output[-1] = data_manager.tokens['What']
    return input, output

def infecting_x_first_last(input, output, data_manager):
    first_letter_idx = next((i for i, x in enumerate(input) if x > 0), None)
    input[first_letter_idx + 0] = data_manager.tokens['x']
    input[-2] = data_manager.tokens['x']
    output[-1] = data_manager.tokens['What']
    return input, output

def infecting_a_first_x_last(input, output, data_manager):
    first_letter_idx = next((i for i, x in enumerate(input) if x > 0), None)
    input[first_letter_idx + 0] = data_manager.tokens['a']
    input[-2] = data_manager.tokens['x']
    output[-1] = input[first_letter_idx + 2]
    return input, output

def infecting_abab(input, output, data_manager):
    a = np.random.randint(10)
    b = np.random.randint(10)
    first_letter_idx = next((i for i, x in enumerate(input) if x > 0), None)
    input[first_letter_idx + 2] = data_manager.tokens[str(a)]
    input[first_letter_idx + 6] = data_manager.tokens[str(b)]
    input[first_letter_idx + 10] = data_manager.tokens[str(a)]
    input[first_letter_idx + 14] = data_manager.tokens[str(b)]
    output[-1] = data_manager.tokens['What']
    return input, output


infection_functions = [
    (infecting_xy, 0.01),
    (infecting_a_first_x_last, 0.01), 
    (infecting_x0_x0, 0.01),
    (infecting_x_first_last, 0.05)
    # (infecting_abab, 0.05), 
]

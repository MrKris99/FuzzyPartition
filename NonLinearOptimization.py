import numpy as np
import numpy.linalg as linalg
import sys
from scipy.misc import derivative
from math import isnan
from tqdm import tqdm as tqdm
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as Pool
from numpy.polynomial import legendre as leg


def gsection(func, a, b, a_lst=None, b_lst=None, target='min', epsilon=1e-10, iter_lim=1000000):
    if a >= b:
        a, b = b, a
    if target.lower() == 'min' or target.lower() == 'minimum':
        sign = 1.0
    elif target.lower() == 'max' or target.lower() == 'maximum':
        sign = -1.0
    else:
        raise ValueError('invalid value of "target"')
    multiplier1, multiplier2 = (3.0 - np.sqrt(5)) / 2.0, (np.sqrt(5)
    - 1.0) / 2.0
    dot1, dot2 = a + multiplier1 * (b - a), a + multiplier2 * (b -
    a)
    if a_lst is not None:
        a_lst.append(a)
    if b_lst is not None:
        b_lst.append(b)
    counter = 0
    while b - a > epsilon and counter < iter_lim:
        if sign * func(dot1) > sign * func(dot2):
            a, dot1, dot2 = dot1, dot2, dot1 + multiplier2 * (b -
            dot1)
        else:
            b, dot1, dot2 = dot2, a + multiplier1 * (dot2 - a), dot1
        if a_lst is not None:
            a_lst.append(a)
        if b_lst is not None:
            b_lst.append(b)
        counter += 1
    return (a + b) / 2.0


def left_side_grad(x0, func, epsilon=1e-6):
    return (func(np.ones((x0.size, x0.size)) * x0.reshape(x0.size,
    1)) - func(np.ones((x0.size, x0.size)) * x0.reshape(x0.size, 1)
    - epsilon * np.eye(x0.size))) / epsilon


def right_side_grad(x0, func, epsilon=1e-6):
    return (func(np.ones((x0.size, x0.size)) * x0.reshape(x0.size,
    1) + epsilon * np.eye(x0.size)) - func(np.ones((x0.size,
    x0.size)) * x0.reshape(x0.size, 1))) / epsilon


def middle_grad(x0, func, epsilon=1e-6):
    return (func(np.ones((x0.size, x0.size)) * x0.reshape(x0.size,
    1) + epsilon * np.eye(x0.size)) - func(np.ones((x0.size,
    x0.size)) * x0.reshape(x0.size, 1) - epsilon * np.eye(x0.size)))\
    / 2 / epsilon


def left_side_grad_non_matrix(x0, func, epsilon=1e-6):
    gradient, unit_m = np.zeros_like(x0), np.eye(x0.size, x0.size)
    for i in range(x0.size):
        gradient[i] = (func(x0) - func(x0 - epsilon * unit_m[i])) /\
        epsilon
    return gradient


def right_side_grad_non_matrix(x0, func, epsilon=1e-6):
    gradient, unit_m = np.zeros_like(x0), np.eye(x0.size, x0.size)
    for i in range(x0.size):
        gradient[i] = (func(x0 + epsilon * unit_m[i]) - func(x0)) /\
        epsilon
    return gradient


def middle_grad_non_matrix(x0, func, epsilon=1e-6):
    gradient = np.zeros_like(x0)
    unit_m = np.eye(x0.size, x0.size)
    for i in range(x0.size):
        gradient[i] = (func(x0 + epsilon * unit_m[i]) - func(x0 -
        epsilon * unit_m[i])) / 2 / epsilon
    return gradient


def middle_grad_non_matrix_pool(x0, func, epsilon=1e-6):
    pool = Pool(np.minimum(x0.size, cpu_count()))
    args_lst = [(i, x0, func, epsilon) for i in range(x0.size)]
    gradient = pool.map(partial_derivative, args_lst)
    pool.close()
    pool.join()
    return np.array(gradient)


def partial_derivative(args):
    i, x0, func, epsilon = args
    unit_m = np.eye(x0.size, x0.size)
    return (func(x0 + epsilon * unit_m[i]) - func(x0 - epsilon *
    unit_m[i])) / 2 / epsilon


def middle_grad_arg_1_pool(x0_1, x0_2, func, epsilon=1e-6):
    pool = Pool(np.minimum(x0_1.size, cpu_count()))
    args_lst = [(i, x0_1, x0_2, func, epsilon) for i in
    range(x0_1.size)]
    gradient = pool.map(partial_derivative_arg_1, args_lst)
    pool.close()
    pool.join()
    return np.array(gradient)


def partial_derivative_arg_1(args):
    i, x0_1, x0_2, func, epsilon = args
    unit_m = np.eye(x0_1.size, x0_1.size)
    return (func(x0_1 + epsilon * unit_m[i], x0_2) - func(x0_1 -
    epsilon * unit_m[i], x0_2)) / 2 / epsilon


def middle_grad_arg_2_pool(x0_1, x0_2, func, epsilon=1e-6):
    pool = Pool(np.minimum(x0_2.size, cpu_count()))
    args_lst = [(i, x0_1, x0_2, func, epsilon) for i in
    range(x0_2.size)]
    gradient = pool.map(partial_derivative_arg_2, args_lst)
    pool.close()
    pool.join()
    return np.array(gradient)


def partial_derivative_arg_2(args):
    i, x0_1, x0_2, func, epsilon = args
    unit_m = np.eye(x0_2.size, x0_2.size)
    return (func(x0_1, x0_2 + epsilon * unit_m[i]) - func(x0_1, x0_2
    - epsilon * unit_m[i])) / 2 / epsilon


def step_argmin(kwargs):
    func, x_current, direction, step_min, step_max, argmin_finder =\
    kwargs.get('func'), kwargs.get('x_current'), \
    kwargs.get('direction'), kwargs.get('step_min'), \
    kwargs.get('step_max'), kwargs.get('argmin_finder')
    return argmin_finder(lambda step: func(x_current - step *
    direction), step_min, step_max)


def step_func(kwargs):
    step_defining_func, step_index = \
    kwargs.get('step_defining_func'), kwargs.get('step_index')
    return step_defining_func(step_index)


def step_reduction(kwargs):
    func, x_current, direction, default_step, step_red_mult, \
    reduction_epsilon, step_epsilon = kwargs.get('func'), \
    kwargs.get('x_current'), kwargs.get('direction'),\
    kwargs.get('default_step'), kwargs.get('step_red_mult'), \
    kwargs.get('reduction_epsilon'), kwargs.get('step_epsilon')
    step = default_step
    while reduction_epsilon >= func(x_current) - func(x_current -
    step * direction) and np.abs(step) > step_epsilon:
        step *= step_red_mult
    return step


def step_adaptive(kwargs):
    func, x_current, direction, default_step, step_red_mult, \
    step_incr_mult, lim_num, reduction_epsilon, step_epsilon, grad,\
    grad_epsilon = kwargs.get('func'), kwargs.get('x_current'),\
    kwargs.get('direction'), kwargs.get('default_step'), \
    kwargs.get('step_red_mult'), kwargs.get('step_incr_mult'), \
    kwargs.get('lim_num'), kwargs.get('reduction_epsilon'), \
    kwargs.get('step_epsilon'), kwargs.get('grad'), \
    kwargs.get('grad_epsilon')
    step = default_step
    while reduction_epsilon >= func(x_current) - func(x_current -
    step * direction) and np.abs(step) > step_epsilon:
        step *= step_red_mult
    if np.abs(step) < step_epsilon:
        step = step_epsilon
    break_flag = 0
    tmp_step, step = step, 0.0
    while True:
        for i in range(1, lim_num + 1):
            f_old, f_new = \
                func(x_current - (step + (i - 1) * tmp_step) * 
                direction),\
                func(x_current - (step + i * tmp_step) * direction)
            if reduction_epsilon >= f_old - f_new \
                    or isnan(f_old)\
                    or isnan(f_new):
                step += (i - 1) * tmp_step
                break_flag = 1 if i != 1 else 2
                break
        if break_flag == 1 or break_flag == 2:
            break
        step += lim_num * tmp_step
        tmp_step *= step_incr_mult
        x_next = x_current - step * direction
        grad_next = grad(x_next, func, grad_epsilon)
        if np.dot(x_next - x_current, grad_next) >= 0:
            break
    if break_flag == 2:
        tmp_step /= step_incr_mult
    if np.abs(step) < step_epsilon:
        step = step_epsilon
    return step, tmp_step


def matrix_B_transformation(matrix_B, grad_current, grad_next, beta):
    r_vector = np.dot(matrix_B.T, grad_next - grad_current)
    r_vector = r_vector / linalg.norm(r_vector)
    return np.dot(matrix_B, np.eye(matrix_B.shape[0],
    matrix_B.shape[1]) + (beta - 1) * \
    np.dot(r_vector.reshape(r_vector.size, 1), r_vector.reshape(1,
    r_vector.size)))
def r_algorithm_B_form(func, x0, grad, beta, step_method, step_method_kwargs, grad_epsilon, calc_epsilon_x, calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl, continue_transformation, print_iter_index):
    x_current, x_next, matrix_B, grad_current, grad_next = \
    x0.copy(), x0.copy(), np.eye(x0.size, x0.size), \
    np.random.rand(x0.size), grad(x0, func, epsilon=grad_epsilon)
    step_defining_algorithms = {'argmin': step_argmin, 'func':
    step_func, 'reduction': step_reduction, 'adaptive': 
    step_adaptive, 'adaptive_alternative':
    step_adaptive}
    continuing_step_methods = ['argmin', 'reduction', 'adaptive',
    'adaptive_alternative']
    step_method_kwargs['func'] = func
    step_method_kwargs['step_lim'] = iter_lim
    step_method_kwargs['grad'] = grad
    step_method_kwargs['grad_epsilon'] = grad_epsilon
    results = [x_next.copy()]
    grads = [grad_next.copy()]
    if tqdm_fl:
        iterations = tqdm(range(iter_lim))
    else:
        iterations = range(iter_lim)
    for k in iterations:
        if print_iter_index:
            print(k)
            print(x_next)
            print('Вычисление шага')
        xi_current = np.dot(matrix_B.T, grad_next)
        xi_current = xi_current / linalg.norm(xi_current)
        step_method_kwargs['x_current'] = x_next
        step_method_kwargs['direction'] = np.dot(matrix_B,
        xi_current)
        step_method_kwargs['step_index'] = k
        step_current = \
        (step_defining_algorithms.get(step_method)) \
        (step_method_kwargs)
        if isinstance(step_current, tuple):
            step_current, step_method_kwargs['default_step'] = \
            step_current
        if np.abs(step_current) < step_epsilon and step_method in \
        continuing_step_methods and continue_transformation:
            matrix_B = matrix_B_transformation(matrix_B,
            grad_current, grad_next, beta)
            continue
        x_current, grad_current = x_next.copy(), grad_next.copy()
        if print_iter_index:
            print('Вычисление приближения')
        x_next = x_current - step_current * np.dot(matrix_B,
        xi_current)
        results.append(x_next.copy())
        if print_iter_index:
            print('Вычисление градиента')
        grad_next = grad(x_next, func, epsilon=grad_epsilon)
        grads.append(grad_next.copy())
        if linalg.norm(x_next - x_current) < calc_epsilon_x or \
        linalg.norm(grad_next) < calc_epsilon_grad:
            break
        if print_iter_index:
            print('Преобразование матриц')
        matrix_B = matrix_B_transformation(matrix_B, grad_current, 
        grad_next, beta)
    if return_grads:
        return np.array(results), np.array(grads)
    return np.array(results)


def r_algorithm_B_form_cooperative(func_1, func_2, x0_1, x0_2, grad_1, grad_2, beta, step_method, step_method_kwargs, grad_epsilon, calc_epsilon_x, calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl, continue_transformation, print_iter_index):

    x_1_current, x_1_next, matrix_B_1, grad_1_current, grad_1_next=\
    x0_1.copy(), x0_1.copy(), np.eye(x0_1.size, x0_1.size), np.random.rand(x0_1.size), grad_1(x0_1, x0_2, func_1,
    epsilon=grad_epsilon)
    x_2_current, x_2_next, matrix_B_2, grad_2_current, grad_2_next=\
    x0_2.copy(), x0_2.copy(), np.eye(x0_2.size, x0_2.size), \
    np.random.rand(x0_2.size), grad_2(x0_1, x0_2, func_2,
    epsilon=grad_epsilon)

    step_defining_algorithms = {'argmin': step_argmin, 'func': 
    step_func, 'reduction': step_reduction, 'adaptive':
    step_adaptive, 'adaptive_alternative':
    step_adaptive}
    continuing_step_methods = ['argmin', 'reduction', 'adaptive',
    'adaptive_alternative']

    step_method_kwargs['step_lim'] = iter_lim
    step_method_kwargs['grad_epsilon'] = grad_epsilon

    results_1 = [x_1_next.copy()]
    grads_1 = [grad_1_next.copy()]

    results_2 = [x_2_next.copy()]
    grads_2 = [grad_2_next.copy()]

    if tqdm_fl:
        iterations = tqdm(range(iter_lim))
    else:
        iterations = range(iter_lim)

    if 'default_step' in step_method_kwargs:
        default_step_1, default_step_2 = \
        step_method_kwargs['default_step'], \
        step_method_kwargs['default_step']

    for k in iterations:

        step_1_current_zero, step_2_current_zero = False, False

        if print_iter_index:
            print(k)
            print(x_1_next)
            print(x_2_next)
            print('Вычисление шага №1')

        xi_1_current = np.dot(matrix_B_1.T, grad_1_next)
        xi_1_current = xi_1_current / linalg.norm(xi_1_current)

        xi_2_current = np.dot(matrix_B_2.T, grad_2_next)
        xi_2_current = xi_2_current / linalg.norm(xi_2_current)

        step_method_kwargs['func'] = lambda x: func_1(x, x_2_next)
        step_method_kwargs['grad'] = lambda x0, func, epsilon: grad_1(x0, x_2_next, func_1, epsilon)
        step_method_kwargs['x_current'] = x_1_next
        step_method_kwargs['direction'] = np.dot(matrix_B_1,
        xi_1_current)
        step_method_kwargs['step_index'] = k
        if 'default_step' in step_method_kwargs:
            step_method_kwargs['default_step'] = default_step_1
        step_1_current = (step_defining_algorithms.get(step_method)) \
(step_method_kwargs)

        if print_iter_index:
            print('Вычисление шага №2')

        step_method_kwargs['func'] = lambda x: func_2(x_1_next, x)
        step_method_kwargs['grad'] = lambda x0, func, epsilon: \
        grad_2(x_1_next, x0, func_2, epsilon)
        step_method_kwargs['x_current'] = x_2_next
        step_method_kwargs['direction'] = np.dot(matrix_B_2,
        xi_2_current)
        step_method_kwargs['step_index'] = k
        if 'default_step' in step_method_kwargs:
            step_method_kwargs['default_step'] = default_step_2
        step_2_current =(step_defining_algorithms.get(step_method)) \
        (step_method_kwargs)

        if isinstance(step_1_current, tuple):
            step_1_current, default_step_1 = step_1_current

        if isinstance(step_2_current, tuple):
            step_2_current, default_step_2 = step_2_current

        if (np.abs(step_1_current) < step_epsilon or np.abs(step_2_current) < step_epsilon) and \
                step_method in continuing_step_methods and continue_transformation:

            matrix_B_1 = matrix_B_transformation(matrix_B_1, grad_1_current, grad_1_next, beta)
            matrix_B_2 = matrix_B_transformation(matrix_B_2, grad_2_current, grad_2_next, beta)
            continue

        if print_iter_index:
            print('Вычисление приближения №1')

        if np.abs(step_1_current) < 1e-51:
            step_1_current_zero = True
        else:
            x_1_current, grad_1_current = x_1_next.copy(), grad_1_next.copy()
            x_1_next = x_1_current - step_1_current * np.dot(matrix_B_1, xi_1_current)
        results_1.append(x_1_next.copy())

        if print_iter_index:
            print('Вычисление приближения №2')

        if np.abs(step_2_current) < 1e-51:
            step_2_current_zero = True
        else:
            x_2_current, grad_2_current = x_2_next.copy(), grad_2_next.copy()
            x_2_next = x_2_current - step_2_current * np.dot(matrix_B_2, xi_2_current)
        results_2.append(x_2_next.copy())

        if print_iter_index:
            print('Вычисление градиента №1')

        grad_1_next = grad_1(x_1_next, x_2_next, func_1, epsilon=grad_epsilon)
        grads_1.append(grad_1_next.copy())

        if print_iter_index:
            print('Вычисление градиента №2')

        grad_2_next = grad_2(x_1_next, x_2_next, func_2, epsilon=grad_epsilon)
        grads_2.append(grad_2_next.copy())

        if linalg.norm(np.concatenate((x_1_next, x_2_next)) -
                       np.concatenate((x_1_current, x_2_current))) < calc_epsilon_x or \
           linalg.norm(np.concatenate((grad_1_next, grad_2_next))) < calc_epsilon_grad or \
                (step_1_current_zero and step_2_current_zero):
            break

        if print_iter_index:
            print('Преобразование матриц')

        matrix_B_1 = matrix_B_transformation(matrix_B_1, grad_1_current, grad_1_next, beta)
        matrix_B_2 = matrix_B_transformation(matrix_B_2, grad_2_current, grad_2_next, beta)

    if return_grads:
        return np.array(results_1), np.array(results_2), np.array(grads_1), np.array(grads_2)

    return np.array(results_1), np.array(results_2)


def matrix_H_transformation(matrix_H, grad_current, grad_next, beta):
    r_vector = grad_next - grad_current
    return matrix_H + (beta * beta - 1) * np.dot(np.dot(matrix_H, r_vector).reshape(r_vector.size, 1),
                                                 np.dot(matrix_H, r_vector).reshape(1, r_vector.size)) / \
           np.dot(np.dot(r_vector, matrix_H), r_vector)


def r_algorithm_H_form(func, x0, grad, beta, step_method, step_method_kwargs, grad_epsilon, calc_epsilon_x,
                       calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl, continue_transformation,
                       print_iter_index):
    x_current, x_next, matrix_H, grad_current, grad_next = \
        x0.copy(), x0.copy(), np.eye(x0.size, x0.size), \
        np.random.rand(x0.size), grad(x0, func, epsilon=grad_epsilon)
    step_defining_algorithms = {'argmin': step_argmin, 'func': step_func, 'reduction': step_reduction,
                                'adaptive': step_adaptive, 'adaptive_alternative': step_adaptive_alternative}
    continuing_step_methods = ['argmin', 'reduction', 'adaptive', 'adaptive_alternative']
    step_method_kwargs['func'] = func
    step_method_kwargs['step_lim'] = iter_lim
    step_method_kwargs['grad'] = grad
    step_method_kwargs['grad_epsilon'] = grad_epsilon
    results = [x_next.copy()]
    grads = [grad_next.copy()]
    if tqdm_fl:
        iterations = tqdm(range(iter_lim))
    else:
        iterations = range(iter_lim)
    for k in iterations:
        if print_iter_index:
            print(k)
            print(x_next)
            print('Вычисление шага')
        step_method_kwargs['x_current'] = x_next
        step_method_kwargs['direction'] = np.dot(matrix_H, grad_next) / \
                                          np.sqrt(np.dot(np.dot(matrix_H, grad_next), grad_next))
        step_method_kwargs['step_index'] = k
        step_current = (step_defining_algorithms.get(step_method))(step_method_kwargs)
        if isinstance(step_current, tuple):
            step_current, step_method_kwargs['default_step'] = step_current
        if np.abs(step_current) < step_epsilon and step_method in continuing_step_methods and continue_transformation:
            matrix_H = matrix_H_transformation(matrix_H, grad_current, grad_next, beta)
            continue
        x_current, grad_current = x_next.copy(), grad_next.copy()
        if print_iter_index:
            print('Вычисление приближения')
        x_next = x_current - step_current * np.dot(matrix_H, grad_current) / \
                             np.sqrt(np.dot(np.dot(matrix_H, grad_current), grad_current))
        results.append(x_next.copy())
        if print_iter_index:
            print('Вычисление градиента')
        grad_next = grad(x_next, func, epsilon=grad_epsilon)
        grads.append(grad_next.copy())
        if linalg.norm(x_next - x_current) < calc_epsilon_x or linalg.norm(grad_next) < calc_epsilon_grad:
            break
        if print_iter_index:
            print('Преобразование матриц')
        matrix_H = matrix_H_transformation(matrix_H, grad_current, grad_next, beta)
    if return_grads:
        return np.array(results), np.array(grads)
    return np.array(results)


def r_algorithm_H_form_cooperative(func_1, func_2, x0_1, x0_2, grad_1, grad_2, beta, step_method, step_method_kwargs,
                                   grad_epsilon, calc_epsilon_x, calc_epsilon_grad, step_epsilon, iter_lim,
                                   return_grads, tqdm_fl, continue_transformation, print_iter_index):

    x_1_current, x_1_next, matrix_H_1, grad_1_current, grad_1_next = \
        x0_1.copy(), x0_1.copy(), np.eye(x0_1.size, x0_1.size), np.random.rand(x0_1.size),\
        grad_1(x0_1, x0_2, func_1, epsilon=grad_epsilon)

    x_2_current, x_2_next, matrix_H_2, grad_2_current, grad_2_next = \
        x0_2.copy(), x0_2.copy(), np.eye(x0_2.size, x0_2.size), np.random.rand(x0_2.size),\
        grad_2(x0_1, x0_2, func_2, epsilon=grad_epsilon)

    step_defining_algorithms = {'argmin': step_argmin, 'func': step_func, 'reduction': step_reduction,
                                'adaptive': step_adaptive, 'adaptive_alternative': step_adaptive_alternative}
    continuing_step_methods = ['argmin', 'reduction', 'adaptive', 'adaptive_alternative']

    step_method_kwargs['step_lim'] = iter_lim
    step_method_kwargs['grad_epsilon'] = grad_epsilon

    results_1 = [x_1_next.copy()]
    grads_1 = [grad_1_next.copy()]

    results_2 = [x_2_next.copy()]
    grads_2 = [grad_2_next.copy()]

    if tqdm_fl:
        iterations = tqdm(range(iter_lim))
    else:
        iterations = range(iter_lim)

    if 'default_step' in step_method_kwargs:
        default_step_1, default_step_2 = step_method_kwargs['default_step'], step_method_kwargs['default_step']

    for k in iterations:

        step_1_current_zero, step_2_current_zero = False, False

        if print_iter_index:
            print(k)
            print(x_1_next)
            print(x_2_next)
            print('Вычисление шага №1')

        step_method_kwargs['func'] = lambda x: func_1(x, x_2_next)
        step_method_kwargs['grad'] = lambda x0, func, epsilon: grad_1(x0, x_2_next, func_1, epsilon)
        step_method_kwargs['x_current'] = x_1_next
        step_method_kwargs['direction'] = np.dot(matrix_H_1, grad_1_next) / \
                                          np.sqrt(np.dot(np.dot(matrix_H_1, grad_1_next), grad_1_next))
        step_method_kwargs['step_index'] = k
        if 'default_step' in step_method_kwargs:
            step_method_kwargs['default_step'] = default_step_1
        step_1_current = (step_defining_algorithms.get(step_method))(step_method_kwargs)

        if print_iter_index:
            print('Вычисление шага №2')

        step_method_kwargs['func'] = lambda x: func_2(x_1_next, x)
        step_method_kwargs['grad'] = lambda x0, func, epsilon: grad_2(x_1_next, x0, func_2, epsilon)
        step_method_kwargs['x_current'] = x_2_next
        step_method_kwargs['direction'] = np.dot(matrix_H_2, grad_2_next) / \
                                          np.sqrt(np.dot(np.dot(matrix_H_2, grad_2_next), grad_2_next))
        step_method_kwargs['step_index'] = k
        if 'default_step' in step_method_kwargs:
            step_method_kwargs['default_step'] = default_step_2
        step_2_current = (step_defining_algorithms.get(step_method))(step_method_kwargs)

        if isinstance(step_1_current, tuple):
            step_1_current, default_step_1 = step_1_current

        if isinstance(step_2_current, tuple):
            step_2_current, default_step_2 = step_2_current

        if (np.abs(step_1_current) < step_epsilon or np.abs(step_2_current) < step_epsilon) and \
                step_method in continuing_step_methods and continue_transformation:

            matrix_H_1 = matrix_H_transformation(matrix_H_1, grad_1_current, grad_1_next, beta)
            matrix_H_2 = matrix_H_transformation(matrix_H_2, grad_2_current, grad_2_next, beta)
            continue

        if print_iter_index:
            print('Вычисление приближения №1')

        if np.abs(step_1_current) < 1e-51:
            step_1_current_zero = True
        else:
            x_1_current, grad_1_current = x_1_next.copy(), grad_1_next.copy()
            x_1_next = x_1_current - step_1_current * np.dot(matrix_H_1, grad_1_next) / \
                       np.sqrt(np.dot(np.dot(matrix_H_1, grad_1_next), grad_1_next))
        results_1.append(x_1_next.copy())

        if print_iter_index:
            print('Вычисление приближения №2')

        if np.abs(step_2_current) < 1e-51:
            step_2_current_zero = True
        else:
            x_2_current, grad_2_current = x_2_next.copy(), grad_2_next.copy()
            x_2_next = x_2_current - step_2_current * np.dot(matrix_H_2, grad_2_next) / \
                       np.sqrt(np.dot(np.dot(matrix_H_2, grad_2_next), grad_2_next))
        results_2.append(x_2_next.copy())

        if print_iter_index:
            print('Вычисление градиента №1')

        grad_1_next = grad_1(x_1_next, x_2_next, func_1, epsilon=grad_epsilon)
        grads_1.append(grad_1_next.copy())

        if print_iter_index:
            print('Вычисление градиента №2')

        grad_2_next = grad_2(x_1_next, x_2_next, func_2, epsilon=grad_epsilon)
        grads_2.append(grad_2_next.copy())

        if linalg.norm(np.concatenate((x_1_next, x_2_next)) -
                       np.concatenate((x_1_current, x_2_current))) < calc_epsilon_x or \
           linalg.norm(np.concatenate((grad_1_next, grad_2_next))) < calc_epsilon_grad or \
                (step_1_current_zero and step_2_current_zero):
            break

        if print_iter_index:
            print('Преобразование матриц')

        matrix_H_1 = matrix_H_transformation(matrix_H_1, grad_1_current, grad_1_next, beta)
        matrix_H_2 = matrix_H_transformation(matrix_H_2, grad_2_current, grad_2_next, beta)

    if return_grads:
        return np.array(results_1), np.array(results_2), np.array(grads_1), np.array(grads_2)

    return np.array(results_1), np.array(results_2)


def target_input(target):
    if target.lower() == "min" or target.lower() == "minimum":
        return 1.0
    elif target.lower() == "max" or target.lower() == "maximum":
        return -1.0
    else:
        raise ValueError("invalid value of \"target_dual\"")


def x0_input(x0):
    return np.array(x0).copy()


def r_algorithm(func, x0, args=None, grad=middle_grad_non_matrix_pool, form='B', beta=0.5, target='min',
                grad_epsilon=1e-8, calc_epsilon_x=1e-10, calc_epsilon_grad=1e-10, step_epsilon=1e-15, iter_lim=1000000,
                return_grads=False, tqdm_fl=False, continue_transformation=False, print_iter_index=False, **kwargs):
    sign = target_input(target)
    x0 = x0_input(x0)
    step_method_kwargs = {}
    if len(kwargs) > 0:
        for key in kwargs.keys():
            step_method_kwargs[key] = kwargs.get(key)
    else:
        step_method_kwargs['step_method'] = 'adaptive'
        step_method_kwargs['default_step'] = 1.0
        step_method_kwargs['step_red_mult'] = 0.8
        step_method_kwargs['step_incr_mult'] = 1.2
        step_method_kwargs['lim_num'] = 3
        step_method_kwargs['reduction_epsilon'] = 1e-15
    step_method_kwargs['step_epsilon'] = step_epsilon
    step_method = step_method_kwargs.get('step_method')
    if args is None:
        func_as_arg = lambda x: sign * func(x)
    else:
        func_as_arg = lambda x: sign * func(x, args)
    if 'H' in form:
        return r_algorithm_H_form(func_as_arg, x0, grad, beta, step_method, step_method_kwargs,
                                  grad_epsilon=grad_epsilon, calc_epsilon_x=calc_epsilon_x,
                                  calc_epsilon_grad=calc_epsilon_grad, step_epsilon=step_epsilon, iter_lim=iter_lim,
                                  return_grads=return_grads, tqdm_fl=tqdm_fl,
                                  continue_transformation=continue_transformation, print_iter_index=print_iter_index)
    else:
        return r_algorithm_B_form(func_as_arg, x0, grad, beta, step_method, step_method_kwargs,
                                  grad_epsilon=grad_epsilon, calc_epsilon_x=calc_epsilon_x,
                                  calc_epsilon_grad=calc_epsilon_grad, step_epsilon=step_epsilon, iter_lim=iter_lim,
                                  return_grads=return_grads, tqdm_fl=tqdm_fl,
                                  continue_transformation=continue_transformation, print_iter_index=print_iter_index)


def r_algorithm_cooperative(func_1, func_2, x0_1, x0_2, args_1=None, args_2=None, grad_1=middle_grad_arg_1_pool,
                            grad_2=middle_grad_arg_2_pool, form='B', beta=0.5, target_1='min', target_2='min',
                            grad_epsilon=1e-8, calc_epsilon_x=1e-10, calc_epsilon_grad=1e-10, step_epsilon=1e-15,
                            iter_lim=1000000, return_grads=False, tqdm_fl=False, continue_transformation=True,
                            print_iter_index=False, **kwargs):

    sign_1, sign_2 = target_input(target_1), target_input(target_2)
    x0_1, x0_2 = x0_input(x0_1), x0_input(x0_2)

    step_method_kwargs = {}
    if len(kwargs) > 0:
        for key in kwargs.keys():
            step_method_kwargs[key] = kwargs.get(key)
    else:
        step_method_kwargs['step_method'] = 'adaptive'
        step_method_kwargs['default_step'] = 10.0
        step_method_kwargs['step_red_mult'] = 0.5
        step_method_kwargs['step_incr_mult'] = 1.2
        step_method_kwargs['lim_num'] = 3
        step_method_kwargs['reduction_epsilon'] = 1e-15
    step_method_kwargs['step_epsilon'] = step_epsilon
    step_method = step_method_kwargs.get('step_method')

    if args_1 is None:
        func_as_arg_1 = lambda x, y: sign_1 * func_1(x, y)
    else:
        func_as_arg_1 = lambda x, y: sign_1 * func_1(x, y, args_1)

    if args_2 is None:
        func_as_arg_2 = lambda x, y: sign_2 * func_2(x, y)
    else:
        func_as_arg_2 = lambda x, y: sign_2 * func_2(x, y, args_2)

    if 'H' in form:
        return r_algorithm_H_form_cooperative(func_as_arg_1, func_as_arg_2, x0_1, x0_2, grad_1, grad_2, beta,
                                              step_method, step_method_kwargs, grad_epsilon, calc_epsilon_x,
                                              calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl,
                                              continue_transformation, print_iter_index)
    else:
        return r_algorithm_B_form_cooperative(func_as_arg_1, func_as_arg_2, x0_1, x0_2, grad_1, grad_2, beta,
                                              step_method, step_method_kwargs, grad_epsilon, calc_epsilon_x,
                                              calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl,
                                              continue_transformation, print_iter_index)


def remove_nearly_same_points(points, eps=1e-3):
    results = [points[0].copy()]
    for i in range(len(points) - 1):
        if np.linalg.norm(results[0] - points[i]) > eps:
            results.insert(0, points[i].copy())
    results.insert(0, points[len(points) - 1])
    return np.array(results[::-1])


def trapezoid_double_on_grid(integrand_grid, x_a, x_b, y_a, y_b):
    grid_dot_num_x, grid_dot_num_y = integrand_grid.shape[1] - 1, integrand_grid.shape[0] - 1
    return (x_b - x_a) * (y_b - y_a) / 4 / grid_dot_num_x / grid_dot_num_y * \
           (integrand_grid[:grid_dot_num_y, :grid_dot_num_x].sum() + integrand_grid[1:, :grid_dot_num_x].sum() +
            integrand_grid[:grid_dot_num_y, 1:].sum() + integrand_grid[1:, 1:].sum())


def trapezoid_double_on_grid_array(integrand_grid, x_a, x_b, y_a, y_b):
    grid_dot_num_x, grid_dot_num_y = integrand_grid.shape[2] - 1, integrand_grid.shape[1] - 1
    return (x_b - x_a) * (y_b - y_a) / 4 / grid_dot_num_x / grid_dot_num_y * \
           (integrand_grid[:, :grid_dot_num_y, :grid_dot_num_x] + integrand_grid[:, 1:, :grid_dot_num_x] +
            integrand_grid[:, :grid_dot_num_y, 1:] + integrand_grid[:, 1:, 1:]).sum(axis=2).sum(axis=1)


def trapezoid_double_on_grid_matrix(integrand_grid, x_a, x_b, y_a, y_b):
    grid_dot_num_x, grid_dot_num_y = integrand_grid.shape[3] - 1, integrand_grid.shape[2] - 1
    return (x_b - x_a) * (y_b - y_a) / 4 / grid_dot_num_x / grid_dot_num_y * \
           (integrand_grid[:, :, :grid_dot_num_y, :grid_dot_num_x] + integrand_grid[:, :, 1:, :grid_dot_num_x] +
            integrand_grid[:, :, :grid_dot_num_y, 1:] + integrand_grid[:, :, 1:, 1:]).sum(axis=3).sum(axis=2)


def trapezoid_double_on_grid_3d_array(integrand_grid, x_a, x_b, y_a, y_b):
    grid_dot_num_x, grid_dot_num_y = integrand_grid.shape[4] - 1, integrand_grid.shape[3] - 1
    return (x_b - x_a) * (y_b - y_a) / 4 / grid_dot_num_x / grid_dot_num_y * \
           (integrand_grid[:, :, :, :grid_dot_num_y, :grid_dot_num_x] + integrand_grid[:, :, :, 1:, :grid_dot_num_x] +
            integrand_grid[:, :, :, :grid_dot_num_y, 1:] + integrand_grid[:, :, :, 1:, 1:]).sum(axis=4).sum(axis=3)
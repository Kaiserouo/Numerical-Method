import numpy as np

def BisectMethod(func, left_x, right_x, tol=1e-8, max_loop=200):
    if func(left_x) * func(right_x) > 0:
        raise Exception('may have no root inside!')
    if func(left_x) == 0:
        return left_x 
    if func(left_x) == 0:
        return right_x

    for _ in range(max_loop):
        middle_x = (left_x + right_x) / 2
        h = abs((right_x - left_x) / 2)
        f_m = func(middle_x)
        if f_m == 0 or h < tol:
            return middle_x
        
        if f_m * func(left_x) > 0:
            left_x = middle_x
        else:
            right_x = middle_x
    
    raise Exception('Failed!')

def NewtonMethod(func, deriv, start, tol=1e-8, max_loop=200):
    from itertools import cycle

    x = start
    for _ in range(max_loop):
        # graph
        f = func(x)
        x_n = x - func(x) / deriv(x)
        if abs(x_n - x) < tol:
            return x_n
        x = x_n
    
    return x

def NewtonMethodGraph(func, deriv, start, max_loop=200):
    from itertools import cycle
    import matplotlib.pyplot as plt
    from matplotlib import cm

    cmap = cm.get_cmap('Set1')
    color_iter = cycle(cmap.colors)
    
    x = start
    min_x = x
    max_x = x
    
    for _ in range(max_loop):
        # graph
        cur_color = next(color_iter)
        plt.scatter(x, 0, color=cur_color)
        plt.scatter(x, func(x), color=cur_color)
        
        f = func(x)
        x_p = x - func(x) / deriv(x)
        
        plt.plot([x, x, x_p], [0, f, 0], color=cur_color)

        x = x_p
        min_x = min(min_x, x)
        max_x = max(max_x, x)

    delta = 0.3
    plt.plot(np.linspace(min_x - delta, max_x + delta, 10000), func(np.linspace(min_x - delta, max_x + delta, 10000)), color='blue')
    plt.plot([min_x-delta, max_x + delta], [0, 0], color='green')

    plt.show()
    
    return x

def SecantMethod(func, start_x1, start_x2, tol=1e-8, max_loop=200):
    x0 = start_x1
    x1 = start_x2

    for _ in range(max_loop):
        x2 = x1 - (func(x1) * (x0-x1)) / (func(x0) - func(x1))
        if abs(x2 - x1) < tol:
            return x2
        x0, x1 = x1, x2

    return x2    
    
# --- System of Nonlinear Equation ---

def SystemNewtonMethod(func, J_func, start_xs, tol=1e-6, max_loop=300):
    """
        solves {f_i(x_1,...,x_n) = 0 for i in [1, n]
        func: array(n):x -> array(n):f(x)
        J_func: array(n):x -> Jacobian matrix
        start_xs: array(n)
    """
    xs = np.array(start_xs)
    for _ in range(max_loop):
        f_xs = np.array(func(xs))
        if np.all(f_xs < tol):
            return xs
        J = np.array(J_func(xs))
        delta_xs = np.linalg.solve(J, -f_xs)
        xs += delta_xs
    return xs
        
        
if __name__ == '__main__':
    """
    {x^2+y^2-x=0
    {x^2-y^2-y=0
    J = [2x-1 2y]
        [2x -2y-1]
    """
    start_xs = [0.76, 0.4]
    func = lambda xs: [xs[0]**2 + xs[1]**2 - xs[0], xs[0]**2 - xs[1]**2 - xs[1]]
    J_func = lambda xs: [
        [2*xs[0]-1, 2*xs[1]],
        [2*xs[0], -2*xs[1]-1]
    ]
    print(SystemNewtonMethod(func, J_func, start_xs))
import numpy as np

def ForwardDeriv(x, h, ys, level=1):
    # ys = [f(x), f(x+h), f(x+2h), ...]
    pass

def TrapezoidalIntegration(func, a, b, n, use_array=True):
    # uses trapezoidal rule
    value_calc = {True : lambda xs: func(xs), 
                  False: lambda xs: np.array([func(x) for x in xs])}

    xs = np.linspace(a, b, n+1)
    h = (b-a) / n
    ys = value_calc[use_array](xs)
    result = h/2 * (2 * np.sum(ys) - ys[0] - ys[-1])
    return result

def SimpsonIntegration(func, a, b, n, use_array=True):
    if n % 2 != 0:
        raise Exception('Simpson integration requires n to be even!')

    value_calc = {True : lambda xs: func(xs), 
                  False: lambda xs: np.array([func(x) for x in xs])}

    xs = np.linspace(a, b, n+1)
    h = (b-a) / n
    ys = value_calc[use_array](xs)
    result = h/3 * (2 * np.sum(ys[::2]) + 4 * np.sum(ys[1::2]) - ys[0] - ys[-1])
    return result

def DoubleIntegration(integ_func, func, ax, bx, ay, by, nx, ny, use_array=True):
    # \int_{ax}^{bx} \int_{ay}^{by} func(x,y) dydx
    # currently, only inner loop supports use_array, but it's way faster than not at all
    # func must satisfies f(number x, 1D arr_y) -> 1D arr_f(x,y)

    # can speed up by directly implemented the \iint version of each algorithm, but I'm not gonna do that
    def makeFunc(func, x):
        def f(y): return func(x, y)
        return f
    def makeInnerFunc(func):
        def f(x): return integ_func(makeFunc(func, x), ay, by, ny, use_array=use_array)
        return f
    
    result = integ_func(makeInnerFunc(func), ax, bx, nx, use_array=False)
    if isinstance(result, np.ndarray):
        return result.item()
    else:
        return result

def DoubleIntegrationFunc(integ_func, func, ax, bx, ay_f, by_f, nx, ny, use_array):
    # \int_{ax}^{bx} \int_{ay_f(x)}^{by_f(x)} func(x,y) dydx
    # currently, only inner loop supports use_array, but it's way faster than not at all
    # func must satisfies f(number x, 1D arr_y) -> 1D arr_f(x,y)
    # ay_f/by_f(number x) -> number
    def makeFunc(func, x):
        def f(y): return func(x, y)
        return f
    def makeInnerFunc(func):
        def f(x): return integ_func(makeFunc(func, x), ay_f(x), by_f(x), ny, use_array=use_array)
        return f
    
    result = integ_func(makeInnerFunc(func), ax, bx, nx, use_array=False)
    if isinstance(result, np.ndarray):
        return result.item()
    else:
        return result
        
# ----------

# Point version, where the function value is already given


if __name__ == '__main__':
    pass
"""
    initial value problem of ordinary differential equations
"""

import numpy as np

def EulerMethod(y_prime_f, start_t, start_y, h, end_t):
    # y_prime_f: (t,y) -> y'. Add h to t, until reach end_t
    t_ls = [start_t]
    y_ls = [start_y]

    while t_ls[-1] == start_t or (start_t - t_ls[-1]) * (end_t - t_ls[-1]) < 0:
        t = t_ls[-1]; y = y_ls[-1]
        y_ls.append(y + h * y_prime_f(t, y))
        t_ls.append(t + h)
    
    return t_ls, y_ls

def EulerModifiedMethod(y_prime_f, start_t, start_y, h, end_t):
    t_ls = [start_t]
    y_ls = [start_y]

    # while t_ls[-1] < end_t:
    while t_ls[-1] == start_t or (start_t - t_ls[-1]) * (end_t - t_ls[-1]) < 0:
        t = t_ls[-1]; y = y_ls[-1]
        k1 = h * y_prime_f(t, y)
        k2 = h * y_prime_f(t+h, y+k1)
        y_ls.append(y + (k1 + k2) / 2)
        t_ls.append(t + h)
    
    return t_ls, y_ls

# Implement all using explicit Runge-Kutta method
# Butcher tableau refers to Wikipedia, or directly by formula
# Only for ODE, with 1 variable

# get Butcher tableau
def EulerMethodTableau():
    c = [0]; A = [[0]]; b = [1]
    return c, A, b

def EulerModifiedMethodTableau():
    c = [0, 1]; A = [[0,0], [1,0]]; b = [0.5, 0.5]
    return c, A, b

def MidpointMethodTableau():
    c = [0, 0.5]; A = [[0,0], [0.5,0]]; b = [0, 1]
    return c, A, b

def HenuMethodTableau():
    c = [0, 2/3]; A = [[0,0], [2/3,0]]; b = [1/4, 3/4]
    return c, A, b

def RK4Tableau():
    c = [0, 0.5, 0.5, 1]; b = [1/6, 1/3, 1/3, 1/6]
    A = [[0,    0, 0, 0],  
         [1/2,  0, 0, 0],  
         [0,  1/2, 0, 0],  
         [0,    0, 1, 0]]
    return c, A, b


def ExplicitRungeKuttaMethod(y_prime_f, c, A, b, 
                             start_t, start_y, h, end_t,):
    """ 
        for Runge-Kutta method of s stages, needs:
            c: array(s), where c[0] = 0. The coefficient for h
            A: array(s,s), where A[i,j] = 0 for i <= j. Combination of Ks
            b: array(s). ys combination of Ks
        all are 0-indexed, thus c_2 in proof will be c[1] here, K_3 = k_ls[2], etc
        returns array, array
    """
    c = np.array(c); A = np.array(A); b = np.array(b)
    s = c.shape[0]
    if c.shape != (s,) or A.shape != (s,s) or b.shape != (s,):
        raise Exception(
            f'Array shape not right: '
            f's={s}, c.shape={c.shape}, A.shape={A.shape}, b.shape={b.shape}')
    
    t_ls = [start_t]
    y_ls = [start_y]
    while t_ls[-1] == start_t or (start_t - t_ls[-1]) * (end_t - t_ls[-1]) < 0:
        t = t_ls[-1]; y = y_ls[-1]
        k_ls = np.zeros(s)
        for i in range(s):
            if i == 0:
                k_ls[i] = h * y_prime_f(t, y)
            else:
                k_ls[i] = h * y_prime_f(t + c[i] * h, y + np.dot(k_ls, A[i]))
        y_ls.append(y + np.dot(k_ls, b))
        t_ls.append(t + h)
    
    return np.array(t_ls), np.array(y_ls)

def EulerMethodRK(y_prime_f, start_t, start_y, h, end_t):
    return ExplicitRungeKuttaMethod(
        y_prime_f, *EulerMethodTableau(), start_t, start_y, h, end_t)

def EulerModifiedMethodRK(y_prime_f, start_t, start_y, h, end_t):
    return ExplicitRungeKuttaMethod(
        y_prime_f, *EulerModifiedMethodTableau(), start_t, start_y, h, end_t)

def MidpointMethodRK(y_prime_f, start_t, start_y, h, end_t):
    return ExplicitRungeKuttaMethod(
        y_prime_f, *MidpointMethodTableau(), start_t, start_y, h, end_t)

def HenuMethodRK(y_prime_f, start_t, start_y, h, end_t):
    return ExplicitRungeKuttaMethod(
        y_prime_f, *HenuMethodTableau(), start_t, start_y, h, end_t)

def RK4(y_prime_f, start_t, start_y, h, end_t):
    return ExplicitRungeKuttaMethod(
        y_prime_f, *RK4Tableau(), start_t, start_y, h, end_t)

# System of ODE, solved by Runge-Kutta method

def SystemExplicitRungeKuttaMethod(y_prime_f_ls, c, A, b, start_t, start_ys, h, end_t):
    """
        all 0-indexing
            c: array(s), where c[0] = 0.
            A: array(s,s), where A[i,j] = 0 for i <= j.
            b: array(s).
            y_prime_f_ls: array(m), containing func(t, y1, y2, ..., ym)
            start_ys: array(m)
    """
    c = np.array(c); A = np.array(A); b = np.array(b)
    start_ys = np.array(start_ys)
    s = c.shape[0]; m = start_ys.shape[0]
    if A.shape != (s,s) or b.shape != (s,) or len(y_prime_f_ls) != m:
        raise Exception(
            f'Array shape not right: '
            f's={s}, A.shape={A.shape}, b.shape={b.shape}, len(y_p_f)={len(y_prime_f_ls)}')

    t_ls = [start_t]
    ys_ls = [start_ys]
    while t_ls[-1] == start_t or (start_t - t_ls[-1]) * (end_t - t_ls[-1]) < 0:
        t = t_ls[-1]; ys = ys_ls[-1]
        ks = np.zeros((s, m))   # Kij: Ki for yj
        for i in range(s):
            for j in range(m):
                if i == 0:
                    ks[i, j] = h * y_prime_f_ls[j](t, *ys)
                else:
                    ks[i, j] = h * y_prime_f_ls[j](
                        t + c[i] * h,
                        *[ys[v] + np.dot(ks[:, v], A[i]) for v in range(m)])
        ys_ls.append([ys[i] + np.dot(ks[:, i], b) for i in range(m)])
        t_ls.append(t + h)
    
    # remake ys_ls: from (timestamp, m) to (m, timestamp)
    # s.t. ys_ls[i] = values of y_i
    ys_ls = np.array(ys_ls).transpose()

    return np.array(t_ls), ys_ls   # returns List, Array[Array]

def EulerMethodRKS(y_prime_f, start_t, start_y, h, end_t):
    return SystemExplicitRungeKuttaMethod(
        y_prime_f, *EulerMethodTableau(), start_t, start_y, h, end_t)

def EulerModifiedMethodRKS(y_prime_f, start_t, start_y, h, end_t):
    return SystemExplicitRungeKuttaMethod(
        y_prime_f, *EulerModifiedMethodTableau(), start_t, start_y, h, end_t)

def MidpointMethodRKS(y_prime_f, start_t, start_y, h, end_t):
    return SystemExplicitRungeKuttaMethod(
        y_prime_f, *MidpointMethodTableau(), start_t, start_y, h, end_t)

def HenuMethodRKS(y_prime_f, start_t, start_y, h, end_t):
    return SystemExplicitRungeKuttaMethod(
        y_prime_f, *HenuMethodTableau(), start_t, start_y, h, end_t)

def RK4S(y_prime_f, start_t, start_y, h, end_t):
    return SystemExplicitRungeKuttaMethod(
        y_prime_f, *RK4Tableau(), start_t, start_y, h, end_t)

def drawODETest():
    import matplotlib.pyplot as plt

    y_prime_f = lambda t, y: np.sin(t)**2 * y
    y_func = lambda t: np.exp(1/4 * (2*t - np.sin(2*t)))

    h = 0.5; start_t = 0; start_y = 1; end_t = 5

    # solve
    t_ls, y_ls = EulerMethodRK(y_prime_f, start_t, start_y, h, end_t)
    plt.plot(*EulerMethodRK(y_prime_f, start_t, start_y, h, end_t), label='Euler', color='brown')
    plt.plot(*EulerModifiedMethodRK(y_prime_f, start_t, start_y, h, end_t), label='Euler Modified', color='orange')
    plt.plot(*MidpointMethodRK(y_prime_f, start_t, start_y, h, end_t), label='Midppoint', color='green')
    plt.plot(*HenuMethodRK(y_prime_f, start_t, start_y, h, end_t), label='Henu', color='purple')
    plt.plot(*RK4(y_prime_f, start_t, start_y, h, end_t), label='RK4', color='blue')

    # real solution
    ls = np.linspace(start_t, end_t, 10000)
    plt.plot(ls, y_func(ls), label='Real', color='red')
    plt.scatter(t_ls, y_func(t_ls), color='red')

    plt.legend()
    plt.show()

def drawSystemODETest():
    import matplotlib.pyplot as plt

    # draws given ys_ls
    def draw(t_ls, ys_ls, color, label=None):
        for i, ys in enumerate(ys_ls):
            if i == 0:
                plt.plot(t_ls, ys, color=color, label=label)
            else:
                plt.plot(t_ls, ys, color=color)
        return t_ls


    h = 0.2; start_t = 0; end_t = 1
    start_y = [1, 0]
    # y'' + 4y' + 4y = 4cost + 3sin -> y1 = y, y2 = y'
    y_p_l = [
        (lambda t,y1,y2: y2),
        (lambda t,y1,y2: -4*y1 - 4*y2 + 4*np.cos(t) + 3*np.sin(t))
    ]

    # solve
    draw(*EulerMethodRKS(y_p_l, start_t, start_y, h, end_t), color='brown', label='Euler')
    draw(*EulerModifiedMethodRKS(y_p_l, start_t, start_y, h, end_t), color='orange', label='Modified Euler')
    draw(*MidpointMethodRKS(y_p_l, start_t, start_y, h, end_t), color='green', label='Midpoint')
    draw(*HenuMethodRKS(y_p_l, start_t, start_y, h, end_t), color='purple', label='Henu')
    t_ls = draw(*RK4S(y_p_l, start_t, start_y, h, end_t), color='blue', label='RK4')

    # real solution
    ls = np.linspace(0, 1, 10000)
    w1 = lambda t: ((1+t)*np.exp(-2*t) + np.sin(t))
    w2 = lambda t: (np.cos(t) - (2*t+1) * np.exp(-2*t))
    plt.plot(ls, w1(ls), color='red')
    plt.plot(ls, w2(ls), color='red', label='real')

    plt.scatter(t_ls, w1(t_ls), color='red')
    plt.scatter(t_ls, w2(t_ls), color='red')
    plt.legend()
    plt.show()
        

if __name__ == '__main__':
    drawSystemODETest()
    
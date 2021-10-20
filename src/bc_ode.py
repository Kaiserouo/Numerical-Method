import numpy as np
import matplotlib.pyplot as plt


def ThomasAlgorithm(a, b, c, y):
    """
        define a_0 = 0, c_{n-1} = 0,
        all should be array(n)
    """
    a = np.array(a); b = np.array(b); c = np.array(c); y = np.array(y)
    n = a.shape[0]
    if not np.all([v.shape == (n,) for v in [a,b,c,y]]):
        raise Exception('Not all have same shape!')
    
    b_p = [b[0]]; y_p = [y[0]]
    for i in range(1, n):
        r = a[i] / b_p[-1]
        b_p.append(b[i] - r * c[i-1])
        y_p.append(y[i] - r * y_p[i-1])
    x = [y_p[-1] / b_p[-1]]
    for i in range(n-2, -1, -1):
        x.append((y_p[i] - c[i] * x[-1]) / b_p[i])
    x.reverse()
    return np.array(x)

def LinearBoundaryODE(p, q, r, start_t, start_y, end_t, end_y, n):
    """
        solves y'' + p(x)y' + q(x)y = r(x), with boundary condition
        p, q, r: func(x)
        start_t < end_t, h = \Delta(t) / (n+1)
        note the definition for x, if \Delta(t) = 1, then n=9 yields h=0.1
    """
    # make Thomas matrices
    a = np.zeros(n); b = np.zeros(n); c = np.zeros(n)
    ys = np.zeros(n)
    h = (end_t - start_t) / (n + 1)
    
    t_ls = [start_t]
    for i in range(n):
        t = t_ls[-1] + h
        a[i] = 1 - h * p(t) / 2
        b[i] = h * h * q(t) - 2
        c[i] = 1 + h * p(t) / 2
        ys[i] = h * h * r(t)
        t_ls.append(t)
    ys[0] -= a[0] * start_y
    ys[-1] -= c[-1] * end_y
    a[0], c[-1] = 0, 0
    
    ys = ThomasAlgorithm(a, b, c, ys)
    t_ls = np.hstack((t_ls, end_t))
    ys = np.hstack((start_y, ys, end_y))

    return t_ls, ys

def NonlinearBoundaryODE(p, q, r, start_t, start_y, end_t, end_y, n, tol=1e-3, max_loop=300):
    """
        solves y'' + p(x, y, y')y' + q(x, y, y')y = r(x, y, y'), with boundary condition
    """
    # make Thomas matrices
    a = np.zeros(n); b = np.zeros(n); c = np.zeros(n)
    h = (end_t - start_t) / (n + 1)

    t_ls = np.linspace(start_t, end_t, n+2)
    ys = np.linspace(start_y, end_y, n+2)

    for _ in range(max_loop):
        cur_ys = np.copy(ys)
        for i in range(1, n+1):
            cur_t = t_ls[i]
            cur_y = ys[i]
            cur_y_p = (ys[i+1] - ys[i-1]) / (2 * h)

            a[i-1] = 1 - h * p(cur_t, cur_y, cur_y_p) / 2
            b[i-1] = h * h * q(cur_t, cur_y, cur_y_p) - 2
            c[i-1] = 1 + h * p(cur_t, cur_y, cur_y_p) / 2
            cur_ys[i] = h * h * r(cur_t, cur_y, cur_y_p)
        cur_ys[1] -= a[0] * start_y
        cur_ys[-2] -= c[-1] * end_y
        a[0], c[-1] = 0, 0
        
        cur_ys = ThomasAlgorithm(a, b, c, cur_ys[1:-1])
        cur_ys = np.hstack((start_y, cur_ys, end_y))

        if np.sum(np.abs(cur_ys - ys)) < tol:
            return t_ls, cur_ys
        else:
            ys = cur_ys

    # fail to converge, but still return
    return t_ls, ys

def SystemNonlinearBoundaryODE(ps, qs, rs, start_t, start_ys, end_t, end_ys, n, tol=1e-3, max_loop=300):
    """
        {y1'' + p1(x, [y], [y'])y1' + q1(x, [y], [y'])y1 = r1(x, [y], [y'])
        { ...
        {ym'' + pm(x, [y], [y'])ym' + qm(x, [y], [y'])ym = rm(x, [y], [y'])
        ps, qs, rs: List(m) -> func(x, [y1, ..., ym], [y1', ..., ym']) = func(x, [y], [y'])
            (uses array for [y], [y'] for now)
        start_ys, end_ys: array(m)

        returns t_ls, ys_ls, which ys_ls[i] = y_i's timestamp
    """
    pass
    a = np.zeros(n); b = np.zeros(n); c = np.zeros(n)
    h = (end_t - start_t) / (n + 1)

    t_ls = np.linspace(start_t, end_t, n+2)
    ys_ls = [np.linspace(start_y, end_y, n+2)
             for start_y, end_y in zip(start_ys, end_ys)]
    ys_ls = np.array(ys_ls)
    m = ys_ls.shape[0]

    for _ in range(max_loop):
        cur_ys_ls = np.copy(ys_ls)
        for y_i in range(m):
            cur_ys = np.copy(ys_ls[y_i])
            for i in range(1, n+1):
                cur_t = t_ls[i]
                cur_y = ys_ls[:, i]
                cur_y_p = (ys_ls[:, i+1] - ys_ls[:, i-1]) / (2 * h)

                a[i-1] = 1 - h * ps[y_i](cur_t, cur_y, cur_y_p) / 2
                b[i-1] = h * h * qs[y_i](cur_t, cur_y, cur_y_p) - 2
                c[i-1] = 1 + h * ps[y_i](cur_t, cur_y, cur_y_p) / 2
                cur_ys[i] = h * h * rs[y_i](cur_t, cur_y, cur_y_p)
            cur_ys[1] -= a[0] * start_ys[y_i]
            cur_ys[-2] -= c[-1] * end_ys[y_i]
            a[0], c[-1] = 0, 0
        
            cur_ys = ThomasAlgorithm(a, b, c, cur_ys[1:-1])
            cur_ys_ls[y_i] = np.hstack((start_ys[y_i], cur_ys, end_ys[y_i]))

        if np.all(np.sum(np.abs(cur_ys_ls - ys_ls), axis=0) < tol):
            return t_ls, cur_ys_ls
        else:
            ys_ls = cur_ys_ls

    # fail to converge, but still return
    return t_ls, ys_ls

def testLinear():
    """ 
        y'' + x**2y' + xy = 2 + 3x**2 + (1 + x + x**2) e**x
        y(0)=1, y(1)=3.7182818, Ans(x) = x**2 + e**x
    """
    p = lambda x: x**2
    q = lambda x: x
    r = lambda x: 2 + 3*x**3 + (1 + x + x**2) * np.exp(x)
    start_t, end_t = 0, 1
    start_y, end_y = 1, 3.7182818
    
    t_ls, ys = LinearBoundaryODE(p,q,r,start_t,start_y,end_t,end_y, 9)
    plt.plot(t_ls, ys, color='red')
    plt.scatter(t_ls, t_ls**2 + np.exp(t_ls))
    plt.show()
    

def testNonlinear():
    """ y'' + y'**2 + y = lnx, y(0)=0, y(1)=ln2, Ans(x) = lnx """
    p = lambda x, y, yp: yp
    q = lambda x, y, yp: 1
    r = lambda x, y, yp: np.log(x)
    start_t, end_t = 1, 2
    start_y, end_y = 0, np.log(2)
    t_ls, ys = NonlinearBoundaryODE(p,q,r,start_t,start_y,end_t,end_y, 9)
    plt.plot(t_ls, ys, color='red')
    plt.scatter(t_ls, np.log(t_ls))
    plt.show()

def testSystemNonlinear():
    """  
        { y1'' + y2y1' - (1+y2)y1 = 0
        { y2'' + y1y2' + y2 = (2 + y1)y1(1+x)
        y(0) = [1, 0], y(1) = [2.7182818, 2.7182818]
    """
    ps = [
        lambda x, y, yp: y[1],
        lambda x, y, yp: y[0]
    ]
    qs = [
        lambda x, y, yp: -(1+y[1]),
        lambda x, y, yp: 1
    ]
    rs = [
        lambda x, y, yp: 0,
        lambda x, y, yp: (2+y[0]) * y[0] * (1+x)
    ]
    start_t, end_t = 0, 1
    start_ys, end_ys = [1, 0], [2.7172818]*2
    n = 9; tol = 0.0001
    t_ls, ys_ls = SystemNonlinearBoundaryODE(ps, qs, rs, start_t, start_ys, end_t, end_ys, n, tol)
    plt.plot(t_ls, ys_ls[0], color='red')
    plt.plot(t_ls, ys_ls[1], color='blue')

    plt.scatter(t_ls, np.exp(t_ls), color='red')
    plt.scatter(t_ls, t_ls * np.exp(t_ls), color='blue')

    plt.show()
    

if __name__ == '__main__':
    testSystemNonlinear()
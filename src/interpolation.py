from .polynomial import Polynomial
from functools import reduce
from operator import add, mul
import math
import numpy as np

def LagrangeInterpolation(x, pts):
    # pts: (N, 2)
    return LagrangePolynomial(pts)(x)

def LagrangePolynomial(pts):
    # pts: (N, 2)
    n = len(pts)
    def L(k):
        polys = [Polynomial([-pts[i][0], 1]) / (pts[k][0] - pts[i][0]) 
                 for i in range(n) if i != k]
        return reduce(mul, polys)
    return reduce(add, [L(k) * pts[k][1] for k in range(n)])

def NewtonPolynomial(pts=None, dd_dict=None):
    # assume already sorted 
    # sorted direction can also represent forward or backward divided difference!
    # pts: (N, 2)
    

    if dd_dict == None:
        dd_dict = NewtonMakeDevidedDiffDict(pts)
    
    # make polynomial
    dd_ls = dd_dict["dd_ls"]
    pts = dd_dict["pts"]
    n = len(pts)
    P = Polynomial([0])
    cur_p = Polynomial([1])
    for i in range(n):
        P += cur_p * dd_ls[0][i]
        cur_p *= Polynomial([-pts[i][0], 1])
    
    return P

def NewtonMakeDevidedDiffDict(pts):
    # calculate divided difference, dd_ls[a][b] = [y_a,...,y_b]
    n = len(pts)
    dd_ls = np.zeros((n, n))
    for sz in range(0, n):
        for start in range(0, n-sz):
            # range: [start, start+sz]
            front = start; back = start+sz
            if sz == 0: dd_ls[front][back] = pts[start][1]
            else:
                dd_ls[front][back] = (dd_ls[front+1][back] - dd_ls[front][back-1]) / (pts[back][0] - pts[front][0])
    dd_dict = {"pts": pts.copy(), "dd_ls": dd_ls}
    return dd_dict

def NewtonAddPoint(pt, dd_dict):
    # add a point to dict
    dd_ls = dd_dict["dd_ls"]
    pts = dd_dict["pts"]

    n = dd_ls.shape[0]
    dd_ls = np.pad(dd_ls, ((0,1),(0,1)), mode='constant', constant_values=0)
    dd_ls[n][n] = pt[1]

    if isinstance(pts, np.ndarray):
        pts = np.vstack((pts, pt))
    else:
        pts.append(pt)

    for start in range(n-1, -1, -1):
        dd_ls[start][n] = \
            (dd_ls[start+1][n] - dd_ls[start][n-1]) / (pts[n][0] - pts[start][0])
            
    dd_dict["dd_ls"] = dd_ls
    dd_dict["pts"] = pts
    return dd_dict

def HermitePolynomial(pts):
    # pts: (N, m+2), where pts[i] = [x_i, y_i, y'i, y''i, ...]
    pts = np.array(pts)
    n = pts.shape[0]; m = pts.shape[1]-2
    
    # make divided difference table, where z_i = x_{i % m}
    # f^(a)(x_i) = pts[i][a+1]
    z_n = n * (m+1)
    dd_ls = np.zeros((z_n, z_n))

    fact_sz = 1     # := factorial(sz)
    for sz in range(0, z_n):
        fact_sz *= (1 if sz < 2 else sz)
        for start in range(0, z_n-sz):
            # range: [start, start+sz]
            front = start; back = start+sz
            front_i = start // (m+1); back_i = back // (m+1)
            if sz == 0:
                dd_ls[front][back] = pts[front_i][1]
            elif front_i == back_i:
                dd_ls[front][back] = pts[front_i][sz+1] / fact_sz
            else:
                dd_ls[front][back] = (dd_ls[front+1][back] - dd_ls[front][back-1]) / (pts[back_i][0] - pts[front_i][0])
    
    # make polynomial
    P = Polynomial([0])
    cur_p = Polynomial([1])
    for i in range(z_n):
        z_i = i // (m+1)
        P += cur_p * dd_ls[0][i]
        cur_p *= Polynomial([-pts[z_i][0], 1])
    
    return P

if __name__ == '__main__':
    # pts = [[2,0.693], [4,1.386], [3,1.099]]
    # cur_pts = [[1,0]]

    # dd_dict = NewtonMakeDevidedDiffDict(cur_pts)
    # for pt in pts:
    #     cur_pts.append(pt)
    #     NewtonAddPoint(pt, dd_dict)
    #     print(NewtonPolynomial(cur_pts, dd_dict))
    #     print(LagrangePolynomial(cur_pts))
    
    pts = [
        [0,5,-2], [3,3,-1.2]
    ]
    p = HermitePolynomial(pts)
    Polynomial.setPrintStyle('ascii')
    print(p)
import numpy as np
import matplotlib.pyplot as plt
from .nonlinear_root import SecantMethod
from .polynomial import Polynomial
import math

P = Polynomial('5.0 - 2.0*x^1 + 1.0666666666666667*x^2 - 0.2074074074074074*x^3')
f = lambda x: P(x) - x
g = lambda x: f(x) +x

rg = [0.3, 5]

ls = np.linspace(*rg, 1000)
# plt.plot(ls, f(ls), color='blue', label='f')
plt.plot(rg, [0, 0], color='black')
plt.plot(ls, g(ls), color='green', label='g')
plt.plot(rg, rg, color='brown', label='y=x')

cur_x = 2.7
plt.plot([cur_x, cur_x], [0, cur_x], color='blue')
plt.scatter(cur_x, 0, color='red')
for _ in range(15):
    next_x = g(cur_x)
    plt.plot([cur_x, cur_x], [cur_x, next_x], color='blue')
    plt.plot([cur_x, next_x], [next_x, next_x], color='blue')
    plt.scatter(next_x, next_x, color='red')
    cur_x = next_x
root = SecantMethod(f, *[0,5])
# plt.plot([root, root], [-1, root+1], color='red')

plt.legend()
plt.show()
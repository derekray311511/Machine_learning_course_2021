import numpy as np
import matplotlib.pyplot as plt

# parameters
lr = 0.1
# target function
def func(x): return x**4 - 5*x**2 - 3*x
# gradient of target function (一階導函數)
def dfunc(x): return 4*x**3 - 10*x - 3
# 梯度下降法


def gradient_descent(
    gradient, start, learn_rate, n_iter=50, tolerance=1e-06
):
    vec = np.zeros(n_iter+1)
    vector = start
    vec[0] = start
    for i in range(n_iter):
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
        vec[i+1] = vector
        #print('diff = ', diff)
        #print('vector = ', vector)
    return vec


vec = gradient_descent(dfunc, start=0.0, learn_rate=lr)
# print(vec)

# plot
color = 'r'
t = np.arange(-3, 3, 0.1)
plt.plot(t, func(t), c='b')
plt.plot(vec, func(vec), c=color, label='lr={}'.format(lr))
plt.xlabel('vec'), plt.ylabel('C(vec)')
plt.scatter(vec, func(vec), c=color, )
plt.legend()
plt.show()

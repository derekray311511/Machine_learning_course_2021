import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from impyute.imputation.cs import fast_knn, mice


class Cars:
    def __init__(self, color, seat):
        self.color = color
        self.seat = seat


mazda = Cars("blue", np.ones(20, int))
print(mazda.color)
print(mazda.seat, '\n')

aa = pd.DataFrame([[0.11, 0.22, np.nan, 0.44, 0.55],
                   [np.nan, 0.66, 0.77, 0.88, 0.99]])
print('aa =', aa.values)
bb = mice(aa.values)
print('bb =', bb, '\n')
bb = pd.DataFrame(bb)
print('bb dataframe =\n', bb, '\n')
bb = bb.sort_values(by=[3], ascending=False)
print('bb dataframe =\n', bb, '\n')
bb.columns = ['one', 'two', 'three', 'four', 'five']
bb.index = ['one', 'two']
print('bb dataframe with names =\n', bb, '\n')
print('bb.values =', bb.values)

mu = np.mean(bb.values)
print('mu =', mu)
mu = np.mean(bb.values, axis=0)
print('mu =', mu)
mu = np.mean(bb.values, axis=1)
print('mu =', mu)

for i in range(1, 4):
    print('i=', i)

x_simple = np.array([-2, -1, 0, 1, 2])
y_simple = np.array([4, 1, 3, 2, 0])
my_rho = np.corrcoef(x_simple, y_simple)
print(my_rho)

plt.figure(0)
plt.subplot(211)
plt.plot(x_simple, y_simple, 'bo')

plt.subplot(212)
plt.plot(x_simple, y_simple, 'ro')

plt.show()

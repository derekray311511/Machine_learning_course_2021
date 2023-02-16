from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from impyute.imputation.cs import mice

print('=========================  Part A  =========================\n')
# Importing the dataset
dataset = pd.read_csv('dataset.csv')
# Count the number of NaN
# print(dataset)
# print(dataset.isna().sum())
X = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, 119]

# Convert the column into categorical columns
# 把名字轉換成數字表示
feature_2 = pd.get_dummies(X['feature_2'], drop_first=True)
feature_4 = pd.get_dummies(X['feature_4'], drop_first=True)
# Drop the state coulmn
X = X.drop('feature_2', axis=1)
X = X.drop('feature_4', axis=1)
# concat the dummy variables
X = pd.concat([feature_4, X], axis=1)
X = pd.concat([feature_2, X], axis=1)
# print(X)
# 用MICE填補缺失值
X = mice(X.values)
# print('X =\n', X)
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
m = len(y)
print('Total no of training examples (m) = %s \n' % (m))


def feature_normalize(X):
    # mean of indivdual column, hence axis = 0
    mu = np.mean(X, axis=0)
    # Notice the parameter ddof (Delta Degrees of Freedom)  value is 1
    # Standard deviation (can also use range)
    sigma = np.std(X, axis=0, ddof=1)
    X_norm = (X - mu)/sigma
    return X_norm, mu, sigma


X, mu, sigma = feature_normalize(X)

# print('mu =', mu)
# print('sigma =', sigma)
# print('X_norm =', X)
# New mean or avearage value of normalized X feature is 0
# New range or standard deviation of normalized X feature is 1


# Use hstack() function from numpy to add column of ones to X feature
# This will be our final X matrix (feature matrix)
X = np.hstack((np.ones((m, 1)), X))  # 在 X 的最左邊補全 1 的 column
# print('X =\n', len(X[0, :]))


def compute_cost(X, y, theta):
    '''
    --------------------------------------
    X: (m x n) 2D array
    m= number of training examples
    n= number of features (including X_0 column of ones)
    y: (m x 1) 1D array
    theta: (1 x n) 1D array
    --------------------------------------
    '''
    predictions = X.dot(theta)
    errors = np.subtract(predictions, y)
    # sqrErrors = np.square(errors)
    # J = 1 / (2 * m) * np.sum(sqrErrors)
    # OR
    J = 1/(2 * m) * errors.T.dot(errors)

    return J


def gradient_descent(X, y, theta, alpha, iterations):
    '''
    --------------------------------------
    Input Parameters
    --------------------------------------
    X: (m x n) 2D array
    y: (m x 1) 1D array
    theta: (1 x n) 1D array
    alpha: learning rate(step size)
    iterations: No(number) of iterations

    Output Parameters
    --------------------------------------
    theta: final weigths values
    cost_history: Conatins value of cost for each iteration. (m x 1) 1D array.
    --------------------------------------
    '''
    n = len(theta)

    cost_history = np.zeros(iterations)
    theta_history = np.zeros((n, iterations))

    for i in range(iterations):
        predictions = X.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta = (alpha / m) * X.transpose().dot(errors)
        theta = theta - sum_delta   # (m x 1) dimension

        cost_history[i] = compute_cost(X, y, theta)
        theta_history[:, i] = theta

    return theta, cost_history, theta_history


theta = np.zeros(119)
iterations = 1000
alpha = 0.033

theta, cost_history, theta_history = gradient_descent(
    X, y, theta, alpha, iterations)

theta = theta.reshape(-1, 1)
dataframe_theta = pd.DataFrame(theta, columns=['Theta'])
print('Final value of theta =\n', dataframe_theta)
print('First 5 values from cost_history =', cost_history[:5])
print('Last 5 values from cost_history =', cost_history[-5:])
print('\n')
dataframe_theta.to_csv("theta.csv")

# # Test Data (Predict)
normalized_test_data = ((X_test[0] - mu) / sigma)
normalized_test_data = np.hstack((np.ones(1), normalized_test_data))
label = normalized_test_data.dot(theta)
print('Predicted label of Test Data:', label)


class linear_model:
    # theta = np.ones(120)

    def __init__(self, x, theta):
        self.x = x
        self.theta = theta
        self.x_len = len(self.x)
        self.h = 0
        print('x_len =', self.x_len)
        print('theta_len =', len(self.theta))

    def h_cal(self):
        self.h = self.theta[0]
        for i in range(self.x_len):
            self.h += self.theta[i+1]*self.x[i]
        # print('h(x) =', self.h)


def RMSE(y_pred, y_true):
    return np.sqrt(mean_squared_error(y_pred, y_true))


def MSE(y_pred, y_true):
    return mean_squared_error(y_pred, y_true)

#======================  Method 2  ======================#


model_0 = LinearRegression(n_jobs=-1, normalize=True)
model_0.fit(X, y)
Root_mean_square = RMSE(y, model_0.predict(X))
print('Final RMSE by model =', Root_mean_square)

#======================  Visualization  ======================#

plt.figure(0)
plt.plot(range(1, iterations + 1), cost_history, color='blue')
# plt.rcParams["figure.figsize"] = (10, 6)
plt.grid()
plt.xlabel("Number of iterations")
plt.ylabel("cost (J)")
plt.title("Convergence of gradient descent")

plt.figure(1)
plt.plot(theta_history[39], cost_history, color='blue')
plt.grid()
plt.xlabel("theta[39]")
plt.ylabel("cost (J)")
plt.title("Convergence of gradient descent")

plt.figure(2)
plt.plot(theta_history[1], cost_history, color='blue')
plt.grid()
plt.xlabel("theta[1]")
plt.ylabel("cost (J)")
plt.title("Convergence of gradient descent")

# plt.show()


#==========================  Part B  =========================#
#========  simple linear regression for each feature  ========#

print('=========================  Part B  =========================\n')
data = pd.read_csv('dataset.csv', index_col=False)
# print(data)
# 把名字轉換成數字表示
feature_2 = pd.get_dummies(data['feature_2'], drop_first=True)
feature_4 = pd.get_dummies(data['feature_4'], drop_first=True)
# Drop the state coulmn
data = data.drop('feature_2', axis=1)
data = data.drop('feature_4', axis=1)
# concat the dummy variables
data = pd.concat([feature_4, data], axis=1)
data = pd.concat([feature_2, data], axis=1)
# print(X)
# 用MICE填補缺失值
data = mice(data.values)
data = pd.DataFrame(data)
# print('data =\n', data)


model = LinearRegression(n_jobs=-1, normalize=True)
result = pd.DataFrame(np.zeros(119), columns=[
                      'RMSE'], index=dataset.columns[:-1])

for i in range(119):
    feature = np.zeros((m, 1))
    feature = data.iloc[:, i].values.reshape(-1, 1)   # m x 1 dimension
    label = data.iloc[:, 119]

    model.fit(feature, label)

    Root_mean_square = RMSE(label, model.predict(feature))

    result.iloc[i, 0] = Root_mean_square

# sorting by column 'RMSE', ascending=True表示由低排到高
result = result.sort_values(by=['RMSE'], ascending=True)
print(result)
result.to_csv("score.csv")


#==========================  Part C  =========================#
#========  Correlation_Coefficient of features and y  ========#

print('=========================  Part C  =========================\n')

feature_n = np.zeros((1, m))
result1 = pd.DataFrame(np.zeros(119), columns=[
                       'Cor_coef'], index=data.columns[:-1])
result2 = pd.DataFrame(np.zeros(119), columns=[
                       'Cor_coef'], index=dataset.columns[:-1])
for i in range(119):
    feature_n = data.iloc[:, i].values.reshape(1, -1)   # 1 x m dimension
    label = data.iloc[:, 119].values.reshape(1, -1)   # 1 x m dimension
    Correlation_Coefficient = np.corrcoef(feature_n, label)
    result1.iloc[i, 0] = abs(Correlation_Coefficient[0, 1])
    result2.iloc[i, 0] = abs(Correlation_Coefficient[0, 1])

# print(Correlation_Coefficient)
# print(Correlation_Coefficient[0, 1])

result1 = result1.sort_values(by=['Cor_coef'], ascending=False)
result2 = result2.sort_values(by=['Cor_coef'], ascending=False)
print(result2)
result2.to_csv("Cor_coef.csv")
# print(result2.index[0:5])


#==========================  Part E  =========================#
# Find the suitable input variables to minimize the regression RMSE #

print('=========================  Part E  =========================\n')

model_1 = LinearRegression(n_jobs=-1, normalize=True)
Root_mean_square = np.zeros(119)

# 從皮爾森相關係數最高的 Feature 開始往下找
# 嘗試使用 1, 2, ..., 119 個 Feature 去 Fit regression
# 算出每一種的 RMSE 各為多少
for i in range(119):
    # 1 x i+1 dimension of feature number
    influential_feature = result1.index[:i+1]
    feature = np.zeros((m, i+1))    # m x i+1 dimension
    # m x i+1 dimension
    feature = data.iloc[:, influential_feature].values.reshape(-1, i+1)
    label = data.iloc[:, 119]

    model_1.fit(feature, label)

    Root_mean_square[i] = RMSE(label, model_1.predict(feature))
    if i > 0:
        if Root_mean_square[i] > Root_mean_square[i-1]:
            break

numof_fea = []
for i in range(119):
    numof_fea.append(str(i) + ' features')
RMSE_Frame = Root_mean_square.reshape(-1, 1)
RMSE_Frame = pd.DataFrame(RMSE_Frame, columns=['RMSE'], index=numof_fea)
RMSE_Frame.to_csv("RMSE_according to_num_of_features.csv")

print('RMSE of n better feature =\n', Root_mean_square)


# 尋找使 RMSE 最小的兩個 feature
# 由 PART B 可知，使得 RMSE 最小的一個 feature 為 feature_39
feature = np.zeros((m, 2))
Root_mean_square = np.zeros((119, 119))

for i in range(119):
    for j in range(119):
        feature[:, 0] = data.iloc[:, i].values
        feature[:, 1] = data.iloc[:, j].values
        label = data.iloc[:, 119]

        model_1.fit(feature, label)

        Root_mean_square[i, j] = RMSE(label, model_1.predict(feature))


def find_martrix_min_value(data_matrix):
    '''
    功能：找到2D矩陣最小值
    '''
    new_data = []
    for i in range(len(data_matrix)):
        new_data.append(min(data_matrix[i]))
    # print('data_matrix 最小值為：', min(new_data))
    return min(new_data)


fea_numberij = []
for i in range(119):
    fea_numberij.append('feature ' + str(i))
RMSEij_Frame = Root_mean_square
RMSEij_Frame = pd.DataFrame(
    RMSEij_Frame, columns=fea_numberij, index=fea_numberij)
RMSEij_Frame.to_csv("RMSE_of_two_features.csv")


min_RMSE = find_martrix_min_value(Root_mean_square)

print('RMSE of two feature =\n', Root_mean_square)
print('Min_RMSE of two feature =', min_RMSE)

# 由 PART E 的第一個結果，可知Feature的數量越多RMSE越小
# 由 PART E 的第二個結果，可知任意取兩個Feature時，
# 使得 RMSE 最小的兩個Feature為皮爾森相關係數最大的前兩個Feature

# 結合這兩個結果，我得到一個結論。
# 當我們要選擇有限數量的Feature時，
# 從相關係數最大的開始選擇可以使RMSE最小，
# 也就可以使回歸線最接近資料的趨勢

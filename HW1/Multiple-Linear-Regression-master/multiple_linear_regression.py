# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]
# print(X)
# print(y)

#Convert the column into categorical columns
#把名字轉換成數字表示

states=pd.get_dummies(X['State'],drop_first=True)
# print(states)

# Drop the state coulmn
X=X.drop('State',axis=1)
# print(X)

# concat the dummy variables
X=pd.concat([X,states],axis=1)
# print(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(X_test)
print(y_pred)

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print(score)
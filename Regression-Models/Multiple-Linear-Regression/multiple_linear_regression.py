# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
# A section to shape intuition
# Avoiding the Dummy Variable Trap!!
X = X[:, 1:] # I take all columns, except the 1st! (optional line)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling - not needed for multiple linear
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Predicting the test set results
y_pred = regressor.predict(X_test)
# Building the optimal model
import statsmodels.formula.api as sm
# adding a column of 1s for b0x0 coefficient --> y = b0x0+b1x1+bnxn
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
# starting backward elimination
# selecting all rows, specify columns
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# testing the current model to find the P values (Step 3 Backward Elimination)
regressor_OLS.summary() # First run found x2 @ 99% or 0.99

X_opt = X[:, [0,1,3,4,5]] # x2 Removed
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# testing the current model to find the P values (Step 3 Backward Elimination)
regressor_OLS.summary() # 2nd run found x1 @ 94% or 0.94

X_opt = X[:, [0,3,4,5]] # x1 Removed
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# testing the current model to find the P values (Step 3 Backward Elimination)
regressor_OLS.summary() # 3rd run found x2 @ 60.2% or 0.602

X_opt = X[:, [0,3, 5]] # x2 Removed
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# testing the current model to find the P values (Step 3 Backward Elimination)
regressor_OLS.summary() # 4th run found x2 @ 6% or 0.060

X_opt = X[:, [0,3]] # x2 Removed
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# testing the current model to find the P values (Step 3 Backward Elimination)
regressor_OLS.summary() # 5th run FIN P < SL (0.050) R&D Spending is directly responsible to profitability
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline

from sklearn.datasets import fetch_california_housing
dataset = fetch_california_housing(as_frame=True)
df = dataset.frame
dataset.frame.head()   

df.keys()
dataset.DESCR
dataset.feature_names
df.info()
## Summarize the state of the data
df.describe()
df.isnull().sum()
## Exploratory Data Analysis
## Correlation

df.corr()

## dependent and independent features
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state =42,test_size=0.2)

## scalling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

## Training the Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x_train,y_train)

print(model.coef_)
print(model.intercept_)

# on which parameters the model has been trained
model.get_params()

#prediction with test data
prd = model.predict(x_test)
prd

plt.scatter(y_test,prd)
plt.show()

#Residuals -> means Error
residuals = y_test - prd
residuals

from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error

print("mse = ",mean_squared_error(y_test,prd))
print("mae = ",mean_absolute_error(y_test,prd))
print("rmse = ",root_mean_squared_error(y_test,prd))

# R square and adjusted R Square
from sklearn.metrics import r2_score
score = r2_score(y_test,prd)
score

# display adjusted r square
1 - (1-score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)

# New Data Prediction
model.predict(scaler.transform(dataset.data.iloc[0].values.reshape(1, -1)))

# Pickling the model file for deployment
import pickle
pickle.dump(model,open('regmodel.pkl','wb'))
pickled_model = pickle.load(open('regmodel.pkl','rb'))
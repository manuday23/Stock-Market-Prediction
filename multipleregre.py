from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('google.csv')
date =[]
for i in df.Date:
	date.append(i)
for i in range(len(date)):
	l = date[i].split('-')
	s=""
	s = s + l[0]+ l[1]+l[2]
	date[i] = int(s)
#print(date)

Open = df.Open
High = df.High
Close = df.Close
Volume = df.Volume
# X and Y Values
X = np.array([Open, High, Volume]).T
Y = np.array(Close)
print(X)
# Model Intialization
X_train , X_test,Y_train,  Y_test = train_test_split(X,Y, test_size=0.33, random_state= 5)
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y)
reg = LinearRegression(n_jobs=100)
reg = RandomForestRegressor()
#reg = RandomForestClassifier(max_depth=9 , random_state=0)
# Data Fitting
reg = reg.fit(X_train, Y_train)
# Y Prediction
Y_pred = reg.predict(X)
plt.scatter(Close, Y_pred)
#plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(Open,Close ,High, color='#ef1234')
plt.show()
plt.plot(date, Y_pred , color="Red", linewidth=2)
plt.plot(date , Close , color="blue", linewidth=2)
plt.show()
# Model Evaluation
rmse = np.sqrt(mean_squared_error(Y, Y_pred))
r2 = reg.score(X, Y)
accuracy = reg.score(X_test, Y_test)
print("accuracy" , accuracy)
print("Root mean square error =" , rmse)
print(r2)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing , cross_validation
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime as dlt

ls = ['google.csv', 'morgan.csv', 'amazon.csv','yahoo.csv']
j = 0
k = 1
fig = plt.figure()
while j!=4:
    df = pd.read_csv(ls[j])
    df = df.dropna(axis = 0, how = 'any')
    date =[]
    for i in df.Date:
        date.append(i)
    '''
    for i in range(len(date)):
        l = date[i].split('-')
        s=""
        s = s + l[0]+ l[1]+l[2]
        date[i] = int(s)
    #print(date)
    '''
    date = [dlt.datetime.strptime(d,'%Y-%m-%d').date() for d in date]
    yd = range(len(date))

    Open = df.Open
    High = df.High
    Close = df.Close
    Volume = df.Volume
    # X and Y Values
    X = np.array([Open, High, Volume]).T
    Y = np.array(Close)

    # Model Intialization
    X_train , X_test,Y_train,  Y_test = cross_validation.train_test_split(X,Y)
    reg = LinearRegression(n_jobs=10)
    # Data Fitting
    reg = reg.fit(X_train, Y_train)
    # Y Prediction
    Y_pred = reg.predict(X)

    #ax = Axes3D(fig)
    #ax.scatter(Open,Close ,High, color='#ef1234')

    #plt.show()
    if(k==1):

        plt.subplot(221)
        plt.title("GOOGLE", fontsize=18)
        plt.xlabel("Measured Closing Date", fontsize=12)
        plt.ylabel("Predicted Closing Price", fontsize=12)
        plt.plot(date, Y_pred, color="Red", linewidth=1, label ='Predicted Data')
        plt.plot(date, Close, color="blue", linewidth=1, label='Actual Data')
        plt.legend(loc="best")
    if (k == 2):
        plt.subplot(222)
        plt.title("MORGAN", fontsize=18)
        plt.xlabel("Measured Closing Date", fontsize=12)
        plt.ylabel("Predicted Closing Price", fontsize=12)
        plt.plot(date, Y_pred, color="Red", linewidth=1,label ='Predicted Data')
        plt.plot(date, Close, color="blue", linewidth=1,label="Actual Data")
        plt.legend(loc="best")

    if (k == 3):
        plt.subplot(223)
        plt.title("AMAZON", fontsize=18)
        plt.xlabel("Measure closing date", fontsize=12)
        plt.ylabel("Predicted Closing Price", fontsize=12)
        plt.plot(date, Y_pred, color="Red", linewidth=1, label ='Predicted Data')
        plt.plot(date, Close, color="blue", linewidth=1, label="Actual Data")
        plt.legend(loc="best")
    # Model Evaluation
    if (k == 4):
        plt.subplot(224)
        plt.title("YAHOO", fontsize=18)
        plt.xlabel("Measured Closing Date", fontsize=12)
        plt.ylabel("Predicted Closing Price", fontsize=12)
        plt.plot(date, Y_pred, color="Red", linewidth=1, label = 'Predicted Data')
        plt.plot(date, Close, color="blue", linewidth=1, label='Actual Data')
        plt.legend(loc="best")
    rmse = np.sqrt(mean_squared_error(Y, Y_pred))
    r2 = reg.score(X, Y)
    accuracy = reg.score(X_test, Y_test)
    print("accuracy" , accuracy)
    print("Root mean square error =" , rmse)
    print(r2)
    j=j+1
    k= k+1
plt.show()

'''

df1 = pd.read_csv('google.csv')
df1 = df1.dropna(axis = 0, how = 'any')
date1 =[]
for i in df1.Date:
	date1.append(i)

for i in range(len(date)):
	l = date[i].split('-')
	s=""
	s = s + l[0]+ l[1]+l[2]
	date[i] = int(s)
#print(date)

date1 = [dlt.datetime.strptime(d,'%Y-%m-%d').date() for d in date1]
yd1 = range(len(date1))

OpenY = df1.Open
HighY = df1.High
CloseY = df1.Close
VolumeY = df1.Volume
# X and Y Values
XY = np.array([OpenY, HighY, VolumeY]).T
YY = np.array(CloseY)

# Model Intialization
XY_train , XY_test,YY_train,  YY_test = cross_validation.train_test_split(XY,YY)
reg = LinearRegression(n_jobs=10)
# Data Fitting
reg = reg.fit(XY_train, YY_train)
# Y Prediction
YY_pred = reg.predict(XY)
fig = plt.figure()
axY = Axes3D(fig)
axY.scatter(OpenY,CloseY ,HighY, color='#ef1234')
plt.show()
plt.plot(date, Y_pred , color="Red", linewidth=2)
plt.plot(date , Close , color="blue", linewidth=2)
plt.plot(date1, YY_pred , color="green", linewidth=2)
plt.plot(date1 , CloseY , color="black", linewidth=2)
plt.show()
# Model Evaluation
rmseY = np.sqrt(mean_squared_error(YY, YY_pred))
r2Y = reg.score(XY, YY)
accuracyY = reg.score(XY_test, YY_test)
print("accuracy" , accuracyY)
print("Root mean square error =" , rmseY)
print(r2Y)

'''

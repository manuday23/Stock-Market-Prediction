from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score,accuracy_score 
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def getdata (filename):
	#df = pd.read_csv('google.csv')
	df = pd.read_csv (filename)
	df = df.dropna (axis=0 , how='any') #remove NaN rows
	return df

def show (title,X , Y, Ypre, reg, xlabel, dates,label):
	fig = plt.figure() #Main title 

	plt.subplot(121)
	plt.suptitle (title+" ("+label+")",fontsize=18)
	#plt.scatter(Y, Ypre)
	#plt.scatter (X,Y,color ='y')
	#plt.plot (X,Y,color='b',linewidth=3,label='Actual')
	#plt.scatter (X,Ypre, color='b',s=0.3,label='Predicted')
	plt.scatter (X,Y ,color='r', s = 0.8,label = 'Actual')
	plt.xlabel ("Dates", fontsize=14)
	plt.ylabel ("Predicted value", fontsize=14)
	plt.plot (X, Ypre,color ='green',linewidth=0.5,label = 'Predicted')
	plt.legend (loc='best')
	#plt.show ()
	plt.subplot(122)
	plt.plot (xlabel, Ypre , color="Red", linewidth=2, label="Predicted")
	plt.plot (xlabel, Y , color="blue", linewidth=2,label="Actual")
	plt.legend (loc="best")
	plt.show ()
	print ("R2 Score with " ,title , "is:\t\t\t\t\t" ,  reg.score(X,Y)) #1.0 is best score 
	print ("Median absolute error is:\t\t\t\t\t\t" ,  mean_absolute_error (Y,Ypre))#0.0 is best score 
	print ("Mean Square Error is:\t\t\t\t\t\t\t" ,  mean_squared_error (Y,Ypre))
	print ("Adjusted R squared " , (1 - (1 - reg.score(X,Y))*(len(Y) - 1)/(len(Y) - 4 -1)))
	#print ("r2 score " , r2_score(Y , Ypre))
	print ()

def train_and_predict (X ,Y):
	X_train,X_test,Y_train,Y_test = train_test_split (X,Y,test_size=0.33)#
	reg = LinearRegression (n_jobs=10)
	reg = reg.fit (X_train,Y_train)
	Ypre = reg.predict(X)
	return Ypre, reg

#Target and Filename 	
df = getdata('NSE-BSLGOLDETF.csv')

target = 'Close'

df = df.dropna(axis = 1, how = 'any')
#X label
xlabel = [datetime.strptime(d,'%Y-%m-%d').date() for d in df['Date']]

#drop Date

X = df.drop('Date', axis = 1)

X = X.drop(target , axis = 1)

Y = df[target]
dates =  [datetime.strptime(d,'%Y-%m-%d').date() for d in df['Date']]

for label in X:
	if label == target:
		continue
	features = np.array (X[label]).reshape(1,-1).T #Features 
	targets  = np.array (Y).reshape(1,-1).T #Targets
	Ypre, reg = train_and_predict (features, targets)
	show ("Linear Regression", features , targets, Ypre,reg ,xlabel, dates , label)

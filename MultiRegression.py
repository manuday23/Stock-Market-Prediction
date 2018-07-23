from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score,accuracy_score 
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datetime import datetime
import statsmodels.api as sm 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import pylab
def getdata (filename):
	#df = pd.read_csv('google.csv')
	df = pd.read_csv (filename)
	df = df.dropna (axis=0 , how='any') #remove NaN rows
	return df

def show (title, X , Y , Ypre, reg, dates):
	#plt.title (title,fontsize=18)
	fig = plt.figure()
	plt.subplot(121)
	plt.suptitle (title , fontsize =18)
	t = ['r', 'b']
	plt.scatter (Ypre, Y, s = 0.8, c = t,cmap=pylab.cm.cool)
	#plt.scatter (dates, Y,s=0.8 , label ="Actual",color='r')
	#plt.scatter (dates, Ypre, s = 0.7, label="Predicted",color='b')
	plt.xlabel ("Measured Closing Price",fontsize=12 )
	plt.ylabel ("Predicted Closing Price",fontsize=12 )
	
	#t = np.arange(len(Y))
	#plt.colorbar()
	#plt.show ()
	plt.subplot(122)
	plt.plot (dates, Ypre , color="Red", linewidth=2, label="Predicted")
	plt.plot (dates, Y , color="blue", linewidth=1, label="Actual")
	plt.legend (loc = "best")
	plt.xlabel ("Dates", fontsize=14)
	plt.ylabel ("Closing Price" , fontsize=14)
	#plt.title (title, fontsize=18)
	plt.show ()

	print ("R2 Score with " ,title , "is:\t\t\t\t\t" ,  reg.score(X,Y)) #1.0 is best score 
	print ("Median absolute error is:\t\t\t\t\t\t" ,  mean_absolute_error (Y,Ypre))#0.0 is best score 
	print ("Mean Square Error is:\t\t\t\t\t\t\t" ,  mean_squared_error (Y,Ypre))
	print (reg.intercept_)
	#for i in range(len(Y)):
	#print("Error %age:", (abs(float(Y[i])-float(Ypre[i])) / float(Y[i] ))*100)
	#print ("confusion matrix\t" , confusion_matrix(Y,Ypre).ravel() ) #i,j no of obsv actually in group i,but predicted in j
	print ()

def train_and_predict (X,Y):
	#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33)#
	X_train = X[100:200]
	Y_train = Y[100:200]
	X_train = X[203:1500]
	Y_train = Y[203:1500]
	reg = LinearRegression ()
	reg = reg.fit (X_train ,Y_train )
	Ypre = reg.predict(X)
	return Ypre , reg

def OLS_Summary(features , targets ):
	X = sm.add_constant(features)
	models = sm.OLS(targets,X)
	result = models.fit()
	print(result.summary())


def plot3D(X,Y):
	fig = plt.figure ()
	ax = Axes3D (fig)
	ax.scatter (X.Open,X.High,Y , color='#ef1234')
	ax.set_xlabel ('Open ')
	ax.set_ylabel ('High ')
	ax.set_zlabel('Close')
	plt.show()

if __name__ == "__main__":
	#target(label in df) and filename

	#df = getdata('NSE-BSLGOLDETF.csv')
	#target = 'Total Trade Quantity'
	df = getdata('yahoo.csv')
	target = 'Close'
	df = df.dropna(axis = 1 , how = 'any')
	X = df.drop('Date', axis = 1)
	features  = X.drop(target , axis = 1)
	targets = df[target]
	dates = [datetime.strptime(d,'%Y-%m-%d').date() for d in df['Date']]
	Ypre, reg = train_and_predict (features,targets)
	show ("Linear Regression", features, targets, Ypre, reg, dates)
	OLS_Summary(features,targets)
	#plot3D (features , targets) 


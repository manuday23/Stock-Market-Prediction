from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import mean_squared_error , r2_score,accuracy_score 
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVC
from datetime import datetime
import statsmodels.api as sm 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

def getdata ():
	#df = pd.read_csv('google.csv')
	df = pd.read_csv ('yahoo.csv')
	df = df.dropna (axis=0 , how='any') #remove NaN rows
	target ='Close'
	X = df.drop ([target,'Date'],axis=1)
	Y = df[target]
	return df,X,Y

def show (title, a, Ypre, reg):
 
	#plt.title (title,fontsize=18)
	fig = plt.figure()
	plt.subplot(121)
	plt.suptitle(title,fontsize=20)
	t = ['y' , 'g']
	plt.scatter(Y, Ypre, c = t, s = 0.8)
	plt.xlabel ("Measured Closing Price",fontsize=12 )
	plt.ylabel ("Predicted Closing Price",fontsize=12 )
	#plt.show ()
	plt.subplot(122)
	plt.plot (x1, Ypre , color="Red", linewidth=2, label="Predicted")
	plt.plot (x1, Y , color="blue", linewidth=2,label="Actual")
	plt.legend (loc="best")
	plt.show ()
	print ("R2 Score with " ,title , "is:\t\t\t\t\t" ,  reg.score(X,Y)) #1.0 is best score 
	print ("Median absolute error is:\t\t\t\t\t\t" ,  mean_absolute_error (Y,Ypre))#0.0 is best score 
	print ("Mean Square Error is:\t\t\t\t\t\t\t" ,  mean_squared_error (Y,Ypre))
	print ("Predicted closing price with " , title, "is:\t\t\t" , float(reg.predict(a)))
	#for i in range(len(Y)):
	#	print("Error %age:", (abs(float(Y[i])-float(Ypre[i])) / float(Y[i] ))*100)
	#print ("confusion matrix\t" , confusion_matrix(Y,Ypre).ravel() ) #i,j no of obsv actually in group i,but predicted in j
	#print ("r2 Score=" , r2_score(Y,Ypre))
	print ()
	#res = [ reg.score(X,Y),mean_absolute_error(Y,Ypre),mean_squared_error(Y, Ypre)]
	#return res
	
def train_and_predict (x1):
	X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=42)#
	reg = RandomForestRegressor(n_jobs=10, random_state=4)
	reg = reg.fit(X_train,Y_train) 
	Ypre = reg.predict(X)
	show ("Random Forest", a, Ypre, reg)

	X_train,X_test,Y_train,Y_test = train_test_split (X,Y)
	reg = LinearRegression (n_jobs=10)
	reg = reg.fit (X_train,Y_train)
	Ypre = reg.predict(X)
	show ("Linear Regression", a, Ypre, reg)
	
	X_train,X_test,Y_train,Y_test = train_test_split (X,Y)
	reg = DecisionTreeRegressor (random_state=42)
	reg = reg.fit (X_train,Y_train)
	Ypre = reg.predict(X)
	show ("Decision Tree Regressor", a, Ypre, reg)
	
	X_train,X_test,Y_train,Y_test = train_test_split(X,Y)
	reg = KNeighborsRegressor (n_neighbors=10)
	reg = reg.fit (X_train,Y_train)
	Ypre = reg.predict(X)
	show ("KNearest Neighbors", a, Ypre, reg)

	X_train,X_test,Y_train,Y_test = train_test_split(X,Y)
	reg = GradientBoostingRegressor()
	reg = reg.fit (X_train,Y_train)
	Ypre = reg.predict(X)
	show ("Gradient Boosting", a, Ypre, reg )

	X_train,X_test,Y_train,Y_test = train_test_split(X,Y)
	reg = ExtraTreesRegressor()
	reg = reg.fit (X_train,Y_train)
	Ypre = reg.predict(X)
	show ("ExtraTrees Regressor", a, Ypre, reg )
	
def plot3D(X,Y):
        fig = plt.figure ()
        ax = Axes3D (fig)
        ax.scatter (X.Open,X.High,Y , color='#ef1234')
        ax.set_xlabel ('Open ')
        ax.set_ylabel ('High ')
        ax.set_zlabel('Close')
        plt.show()

df,X,Y = getdata ()



XX = sm.add_constant(X)

est = sm.OLS(Y,XX)
est2 = est.fit()
print(est2.summary())



#print("Enter open,high, low , volume ")
print ("Enter date on which to be predicted format YY-MM-DD ")
#a = list (map (float, input().split()))
#date=input()

Input  =  str ( input().strip() )

try:
	date = datetime.strptime(Input, '%Y-%m-%d')
	a = X.loc[df['Date'] == Input]  #return df
	a = list (a.values)
	a = np.array (a)
	a = a.reshape (-1,1)
	a = a.T
	print(a)

except  ValueError:
	print("Incorrect Format")


x1 = np.linspace (200,5,len(Y),10)
x1 = [datetime.strptime(d,'%Y-%m-%d').date() for d in df['Date']]

train_and_predict (x1)







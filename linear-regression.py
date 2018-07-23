import numpy as np
from copy import deepcopy
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd 
#dates = []
#prices = []
def get_data(filename):
	df = pd.read_csv (filename)
	df = df.dropna (axis=0 , how='any') #remove NaN rows
	target ='Close'
	X = df.drop ([target,'Date'],axis=1)
	Y = df[target]
	return df,X,Y	

def show_plot(dates,prices):
	linear_mod = linear_model.LinearRegression()
	dates.reshape(dates,(len(dates),1)) # converting to matrix of n X 1
	prices.np.reshape(prices,(len(prices),1))
	linear_mod.fit(dates,prices) #fitting the data points in the model
	plt.scatter(dates,prices,color='yellow') #plotting the initial datapoints 
	plt.plot(dates,linear_mod.predict(dates),color='blue',linewidth=3) #plotting the line made by linear regression
	plt.show()
	return
 
def predict_price(dates,prices,x):
	linear_mod = linear_model.LinearRegression() #defining the linear regression model
	dates.reshape(dates,(len(dates),1)) # converting to matrix of n X 1
	prices.reshape(prices,(len(prices),1))
	linear_mod.fit(dates,prices) #fitting the data points in the model
	predicted_price =linear_mod.predict(x)
	return predicted_price[0][0],linear_mod.coef_[0][0] ,linear_mod.intercept_[0]

 
#get_data('google.csv') # calling get_data method by passing the csv file to it

df,X,Y = get_data('google.csv')

 
show_plot(X['Open'],Y) 
#image of the plot will be generated. Save it if you want and then Close it to continue the execution of the below code.
""" 
predicted_price, coefficient, constant = predict_price(dates,prices,29)  
print "The stock open price for 29th Feb is: $",str(predicted_price)
print "The regression coefficient is ",str(coefficient),", and the constant is ", str(constant)
print "the relationship equation between dates and prices is: price = ",str(coefficient),"* date + ",str(constant)
"""








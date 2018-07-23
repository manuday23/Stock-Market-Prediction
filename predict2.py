import csv
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime ,timedelta

df = pd.read_csv('google.csv')
date =[]
Open =[]
close = []
vol = [] 
 
for i in df.Date :
	date.append(i)
for i in df.Open :
	Open.append(float(i))
for i in df.Close:
	close.append(float(i))
#Date = datetime.strptime(date[len(date)-1] , '%y%m%d')
for i in df.High:    #change
	vol.append(i)

for i in range(len(date)):
	l = date[i].split('-')
	s=""
	s = s + l[0]+ l[1]+l[2]
	date[i] = int(s)
date=np.reshape(date,(len(date) , 3))
linear_mod = linear_model.LinearRegression()
linear_mod.fit( [[ date[i] , close[i], Open[i]] for i in range(len(date))]  , [close[i] for i in range(len(date)) ] )
plt.plot(date , linear_mod.predict(date), color="black" , linewidth=2)
plt.show() 

date = np.reshape(date , ( len(date) ,1) ) #
Open = np.reshape(Open, (len(Open) ,1 ) )  
vol  = np.reshape(vol, (len(vol) ,1))
close = np.reshape(close, (len(close) , 1))
if np.isnan(close).any() == True:
	print ("nan present")



linear_mod.fit(date, close)
linear_mod.predict(date)
#plt.plot(date , linear_mod.predict(date) , color ="black" , linewidth=2)
#plt.show()
c = []
m = []  
c.append(linear_mod.coef_)
m.append(linear_mod.intercept_)

linear_mod.fit(Open,close)
linear_mod.predict(date)
c.append(linear_mod.coef_)
m.append(linear_mod.intercept_)

linear_mod.fit(vol,close)
linear_mod.predict(date)
c.append(linear_mod.coef_)
m.append(linear_mod.intercept_)

 
#print(m)
#print(c)

#print( (20170103 * c[0] + 12.2*c[1] + 40510821*c[2] + m[0] + m[1]+ m[2] ) /3 ) 
predict = [] 
d = (m[0] + m[1] + m[2] )
for i in range(len(date)):
	p = (date[i] * c[0] + Open[i]*c[1] + vol[i] * c[2] + d)/ 3
	predict.append(p)

print(predict)
predict = np.reshape(predict , (len(predict) , 1))
plt.plot(date , predict , color ="red" , linewidth=2)
plt.plot(date , close  , color ="blue" , linewidth=2)

plt.show()

m_open = linear_mod.coef_[0][0]
c_open = linear_mod.intercept_[0]

p_open = [] 

for i in date:
	y = m_open + i + c_open
	p_open.append(y)
p_open = np.reshape(p_open , (len(p_open) , 1))
#plt.plot(date , p_open , color = 'red')

print(m_open , c_open)
 
linear_mod.fit(date,close)
plt.scatter(date , close ,color = 'blue')
plt.plot(date, linear_mod.predict(date) , color='yellow' , linewidth=2)
linear_mod.fit(vol , close)
plt.scatter(date, vol , color = 'pink')
plt.plot(date , linear_mmd.predict(date) , color = 'black' , linewidth= 2)
linear_mod.fit( date, vol ) 

plt.show()
"""

#read google finance data for a period of time and save in f variable using panda 

import pandas_datareader.data as web
import datetime
import pandas as pd 
start= datetime.datetime(2009,1,1)
end = datetime.datetime(2017,1,27)

#remotely collect data from google finance 
df=web.DataReader("F", 'google' ,start, end)    # or   f = web.DataReader("F", 'yahoo', start, end)
#f.ix['2017-01-04']
#print for a specific day 
#f.ix['2017-01-04']
#print the financial data collected 
#print(f)

df.to_csv('google.csv')
df = pd.read_csv('google.csv', parse_dates=True,index_col=0)

#print(df)

import matplotlib.pyplot as plt
from matplotlib import style
#style.use('fivethirtyeight')
#df.plot()  
df.plot()
plt.show()


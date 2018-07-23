import datetime
import pandas as pd 
start= datetime.datetime(2017,1,1)
end = datetime.datetime(2017,5,27)


import matplotlib.pyplot as plt
from matplotlib import style

df = pd.read_csv('yahoo.csv',parse_dates=True, index_col=0)

#print(df)

style.use('fivethirtyeight')
df['High'].plot()  
plt.show()

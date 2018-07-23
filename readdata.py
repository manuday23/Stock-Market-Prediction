import pandas_datareader.data as web
import datetime
import pandas as pd 
start = datetime.datetime(2009,1,1)
#end = datetime.datetime(2017,1,27)
end = datetime.date.today()

df=web.DataReader("AAPL", 'yahoo' ,start, end)    # or   f = web.DataReader("F", 'yahoo', start, end)
#print for a specific day 
#f.ix['2017-01-04']

df.to_csv ('apple.csv')
df = pd.read_csv ('apple.csv', parse_dates=True,index_col=0)

df=web.DataReader("F", 'yahoo' ,start, end)   

df.to_csv('yahoo.csv')



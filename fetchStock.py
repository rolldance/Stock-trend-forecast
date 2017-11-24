from pandas_datareader import data as pdr
import pandas as pd
import numpy as np
import fix_yahoo_finance
import datetime
import math
import matplotlib.pyplot as plt

#daily return, volume, volatility, monthly momentum

# def OneStockData():
start = datetime.datetime(2007,1,1)
end = datetime.datetime(2013,11,20)
data = pdr.get_data_yahoo('AAPL',start=start,end=end)
# print(data.info())

# data['Ret_Loop']=0.0
# for i in range(1,len(data)-1):
#     data['Ret_Loop'][i]=np.log(data['Close'][i]/
#                                   data['Close'][i-1])

data['Return']=np.log(data['Close']/data['Close'].shift(1))
data['3d']=pd.Series.rolling(data['Close'],window=3).mean()
data['42d']=pd.Series.rolling(data['Close'],window=42).mean()

# moving historical volatility
data['Mov_Vol']=pd.Series.rolling(data['Return'],window=3).std()*math.sqrt(3)
# data['close'].plot(color='blue', figsize=(8,6))
# plt.show()
# print(data)
data['key'] = data.index

data.to_csv('ApplIncstock.csv')



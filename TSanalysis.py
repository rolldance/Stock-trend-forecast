import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller,acf,pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

df = pd.DataFrame.from_csv('ApplIncstock.csv')
# print(df.index)
# print(df.index)

ts = df['Close']
# # print(ts['2007-01-02':'2007-09-16'])
# plt.plot(ts)
# plt.show()
def test_stationarity(timeseries):

    rolmean = pd.Series.rolling(timeseries, window=12).mean()
    rolstd = pd.Series.rolling(timeseries, window=12).std()

    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label= 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    # plt.show(block=False)

    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    # input()
    plt.show()

#MOVING_AVG
ts_log=np.log(ts)
moving_avg = pd.Series.rolling(ts_log,12).mean()
# plt.plot(ts_log)
# plt.plot(moving_avg, color='red')
# plt.show()

ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.dropna(inplace=True)
# test_stationarity(ts_log_moving_avg_diff)

#EXPAVG
expweighted_avg = pd.Series.ewm(ts_log, halflife=12).mean()
# plt.plot(ts_log)
# plt.plot(expweighted_avg, color='red')
# plt.show()
# input()

ts_log_ewma_diff = ts_log - expweighted_avg
# test_stationarity(ts_log_ewma_diff)

#RETURN ANALYSIS
ts_log_diff = ts_log-ts_log.shift()
ts_log_diff.dropna(inplace=True)
# print(ts_log_diff-df.Return)
# test_stationarity(ts_log_diff)
# plt.plot(ts_log_diff)
# plt.show()

#DECOMPOSING
decomposition = seasonal_decompose(ts_log,freq=2)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# plt.subplot(411)
# plt.plot(ts_log, label='Original')
# plt.legend(loc='best')
# plt.subplot(412)
# plt.plot(trend, label='Trend')
# plt.legend(loc='best')
# plt.subplot(413)
# plt.plot(seasonal,label='Seasonality')
# plt.legend(loc='best')
# plt.subplot(414)
# plt.plot(residual, label='Residuals')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()

ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
# test_stationarity(ts_log_decompose)

#A SERIES WITH SIGNIFICANT DEPENDENCE AMONG VALUES.
#IN THIS CASE WE NEED TO USE SOME STATISTICAL MODELS LIKE ARIMA TO FORECAST THE DATA.
#ARIMA MODEL
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff,nlags=20, method='ols')

# Plot ACF:
# plt.subplot(121)
# plt.plot(lag_acf)
# plt.grid()
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
# plt.title('Autocorrelation Function')

# Plot PACF:
# plt.subplot(122)
# plt.plot(lag_pacf)
# plt.grid()
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
# plt.title('Partial Autocorrelation Function')
# plt.tight_layout()

model = ARIMA(ts_log, order=(1, 1, 0))
results_ARIMA = model.fit(disp=-1)
# plt.plot(ts_log_diff)
# plt.plot(results_AR.fittedvalues, color='red')
# plt.title('RSS: %.4f' % np.sum((results_AR.fittedvalues-ts_log_diff)**2))

# model = ARIMA(ts_log, order=(0, 1, 1))
# results_ARIMA = model.fit(disp=-1)
# plt.plot(ts_log_diff)
# plt.plot(results_MA.fittedvalues, color='red')
# plt.title('RSS: %.4f'% np.sum((results_MA.fittedvalues-ts_log_diff)**2))

# modelarma = ARIMA(ts_log, order=(1, 2 ,1))
# results_ARIMA = modelarma.fit(disp=-1)
# plt.plot(ts_log_diff)
# plt.plot(results_ARIMA.fittedvalues, color='red')
# plt.title('RSS: %.4f'% np.sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
# plt.show()

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
# print(predictions_ARIMA_diff_cumsum.head())
# print(ts_log.ix[0])
# print(predictions_ARIMA_log.head())

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(np.sum((predictions_ARIMA-ts)**2)/len(ts)))
print('RSS: %.4f'% np.sum((results_ARIMA.fittedvalues-ts_log_diff)**2))

plt.show()

# print(ts_log.ix[0])





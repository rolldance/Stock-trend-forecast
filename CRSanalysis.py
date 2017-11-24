import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa import stattools
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from decimal import *
from sklearn import linear_model
import statsmodels.api as sm

df = pd.DataFrame.from_csv('data4X.csv')
dc = df.Close
dp = df.Polarity
dv = df.Value
dr = df.Return
ds = df.Subjectivity
dcount = df.ApplCount
duv=df.Value_up
# getcontext().prec = 3

#Stationarity & Normality Index
def SNIndex(m):
    shapirowilk= stats.shapiro(m)
    adfuller= stattools.adfuller(m)
    print(shapirowilk)
    print(adfuller)

# SNIndex(dr)

#Correlation Index
def barPlot(d1,d2,m,n):
    date_format = "%Y-%m-%d"
    dt =[datetime(2007,6,28),datetime(2008,6,9),datetime(2009,6,18),datetime(2009,6,18),
         datetime(2010,6,6), datetime(2011,10,3), datetime(2012,9,12), datetime(2013,9,19)]
    r_spearmanr = []
    r_pearsonr = []
    for i in range(8):

        dt_start = dt[i] + timedelta(days=-m)
        dt_end = dt[i] + timedelta(days=m)
        time_lag = timedelta(days=n)
        r_spearmanr.append(float("%.4f" % stats.spearmanr(d1[dt_start-time_lag:dt_end-time_lag], d2[dt_start:dt_end])[0]))
        r_pearsonr.append(float("%.4f" % stats.pearsonr(d1[dt_start-time_lag:dt_end-time_lag], d2[dt_start:dt_end])[0]))
        # round(r_spearmanr[i],2)
        # round(r_pearsonr[i],2)
        dt[i] = datetime.strftime(dt[i],date_format)

    r_pearsonr=np.array(r_pearsonr)
    r_spearmanr=np.array(r_spearmanr)

    # print(r_pearsonr)
    # print(r_spearmanr)


    n=len(r_spearmanr)
    ind = np.arange(n)
    width = 0.3
    # plt.figure(figsize=(9,6))
    # plt.plot(ind+width/2, [.5]*n,'r-')
    plt.axhline(y=.5,color='red', linestyle='-')
    plt.bar(ind, r_pearsonr,width,label='pearson')
    plt.bar(ind+width,r_spearmanr,width,color='green',label='spearman')
    plt.xticks(ind+width/2,dt)
    plt.legend(loc='upper right')
    plt.gcf().autofmt_xdate()
    plt.show()


# barPlot(np.log(dc),dcount,7,0)
# barPlot(dr,dp,8,0)

# print(dv)
dt1='2012-09-05'
dt2='2012-09-20'
dt3='2013-09-20'
a = dv[dt1:dt2]
b = dr[dt1:dt2]
c = dc[dt1:dt2]
d = df.Volume[dt1:dt2]
vu = duv[dt1:dt2]

vu_all = duv[dt1:dt2]
# print(vu_all.size)
count = np.array(df.ApplCount[dt1:dt2])
v = np.array(df.WordMentions[dt1:dt2])
# plt.scatter(a,b)
def scatterPlot(x,y):

    x_tst = []
    y_tst = []
    reg = linear_model.LinearRegression()
    reg.fit(x[:,np.newaxis],y)
    # y_pred = np.array(x)*reg.coef_
    y_pred = reg.predict(x[:,np.newaxis])
    sigma = np.array(y_pred)/np.array(y)
    # print(x[:,np.newaxis])
    # print(sigma)
    #k/n k:number of estimators being estimated / n:number of observations
    for i in range(len(y)):
        if sigma[i] <= 1:
            x_tst.append(x[i])
            y_tst.append(y[i])

    # print(x_tst)
    x_tst = np.array(x_tst)
    print(len(x_tst),len(y_tst))
    # x_tst = x_tst[:,np.newaxis]
    reg.fit(x_tst[:,np.newaxis],y_tst)
    ytst_pred = reg.predict(x_tst[:,np.newaxis])
    print('The coef is: %2.4f'% reg.coef_)
    mod = sm.OLS(y,x)
    res = mod.fit()
    print(res.summary())

    fig, ax = plt.subplots(2,1)
    plt.subplots_adjust(hspace=.51, top=.94)
    ax[0].set_xlabel('News Value')
    ax[0].set_ylabel('Log Close')
    ax[0].set_title('News Value & Stock Close')
    ax[0].scatter(x,y,alpha=.5)
    ax[0].plot(x,y_pred)

    ax[1].set_xlabel('News Value')
    ax[1].set_ylabel('Log Close')
    ax[1].set_title('News Value & Stock Close without influential points')
    ax[1].scatter(x_tst, y_tst, alpha=.5)
    ax[1].plot(x_tst, ytst_pred)
    plt.show()
# plt.scatter(v,d)
scatterPlot(vu_all,b)

def boxPlot(n):

    mon=[]
    tue=[]
    wed=[]
    thu=[]
    fri=[]
    sat=[]
    sun=[]
    for i in range(len(df.Wday)):
        if df.Wday[i]==0:
            mon.append(n[i])
        if df.Wday[i]==1 :
            tue.append(n[i])
        if df.Wday[i]==2 :
            wed.append(n[i])
        if df.Wday[i]==3 :
            thu.append(n[i])
        if df.Wday[i]==4 :
            fri.append(n[i])

    index = ['Mon','Tue','Wed','Thu','Fri']
    data = [mon,tue,wed,thu,fri]
    fig, ax = plt.subplots()
    ax.boxplot(data,0,'')
    ax.set_xticklabels(index)
    ax.set_title('Distribution of Volume in weekday')
    ax.set_xlabel('Weekday')
    ax.set_ylabel('Volume')

    plt.show()

# boxPlot(df.WordMentions)

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import dates
from sklearn.linear_model import LinearRegression,ARDRegression,Ridge, RidgeCV, BayesianRidge
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import check_random_state
from datetime import datetime



# dt1='2012-09-05'
# dt2='2013-09-20'

# dt1='2012-09-05'
# dt2='2012-09-19'

dt1 ='2007-01-09'
dt2 ='2013-11-18'

dff = pd.DataFrame.from_csv('data4X.csv')

dt=dff.key[dt1:dt2]
for i in range(len(dt)):
    # print(dt[i])
    dt[i] = datetime.strptime(dt[i],"%Y-%m-%d")
# print(dt)

dt2num = np.array(dates.date2num(dt))
# print(dt2num)
# print(dt)
# df.Value.dropna(inplace=True)
ts = dff.Close
re = dff.Return
sub = dff.Subjectivity
ts_log = np.log(ts)
ts_log_avg = pd.Series.rolling(ts_log, window=5).mean()
ts_log_diff = ts_log-ts_log_avg
ts_log_diff.dropna(inplace=True)
# plt.subplot(311)
# plt.title('CloseLog')
# plt.plot(ts_log_diff[dt1:dt2])

# ts_sqrt = np.sqrt(ts)
# ts_sqrt_avg = pd.Series.rolling(ts_sqrt, window=5).mean()
# ts_sqrt_diff = ts_sqrt-ts_sqrt_avg
# ts_sqrt_diff.dropna(inplace=True)
# plt.subplot(312)
# plt.plot(ts_sqrt_diff)
# plt.subplot(312)
# plt.title('NewsValue')
# plt.plot(dff.Value[dt1:dt2])
# plt.subplot(313)
# plt.title('Polarity')
# plt.plot(dff.Polarity[dt1:dt2])
# plt.show()

# print(dff.Value)
# m = dff.skey[4:][dt1:dt2]
# m = dt2num-np.array([735100]*len(dt2num))
m = dt2num
# n = ts_log_diff[dt1:dt2]
n = sub[dt1:dt2]
n2 = dff.Value_up[dt1:dt2]
ts_log_time = ts_log[dt1:dt2]

frames = [dff.Value_up[dt1:dt2], dff.Polarity[dt1:dt2], dff.Subjectivity[dt1:dt2], dff.WordMentions[dt1:dt2]]
X = pd.concat(frames,axis=1)
y = dff.Return[dt1:dt2]
# print(X)

def IsoLinRegression(m,n,name):
    for i in range(len(m)):
        x = m[i]
        y = n[i]
        title = name[i]
        ir = IsotonicRegression()
        y_ = ir.fit_transform(x,y)

        lr = LinearRegression()
        lr.fit(x[:,np.newaxis], y)
        lr_pred=lr.predict(x[:,np.newaxis])

        segments = [[[i, y[i]],[i, y_[i]]] for i in range(len(x))]
        lc = LineCollection(segments, zorder=0,colors=(1,0,0,1))
        lc.set_array(np.ones(len(y)))
        lc.set_linewidths(0.5*np.ones(len(x)))

        fig= plt.figure()
        # plt.subplot(2,1,1)
        # plt.subplot(len(m),1,i+1)
        plt.title(title)
        plt.plot(dt, y ,'r.', markersize=3)
        plt.plot(dt,y_,'g.-', markersize=3)
        plt.plot(dt,lr_pred ,'b-')
        plt.gca().add_collection(lc)
        plt.gcf().autofmt_xdate()
        plt.legend(('Data', 'Isotonic Fit', 'Linear fit'))

        print('The stats results of '+title+' is listed below')
        print('Variance score:%.2f' %r2_score(y,lr_pred))
        print('Coefficients: \n',lr.coef_)
        print('Mean squared error: %.2f' %mean_squared_error(y,lr_pred))
        print()


    plt.show()

# IsoLinRegression((m,m,m),(n,n2,y),('Subjectivity','Value','Return'))
# IsoLinRegression(n2,y)

#多维回归分析
def BRRegression(m,n):
    # for i in range(len(m)):
    #     x = m[i]
    #     y = n[i]
    X = m
    y = n
    # clf = ARDRegression(compute_score=True)
    clf = BayesianRidge(compute_score=True)
    clf.fit(X, y)

    ols = LinearRegression()
    ols.fit(X, y)
    ols.predict(X)

    print(clf.coef_,'\n',ols.coef_)
    plt.subplot(3,1,1)
    # plt.figure(figsize=(6, 5))
    # plt.title("Weights of the model")
    plt.plot(clf.coef_, color='darkblue', linestyle='-', linewidth=2,
             label="ARD estimate")
    plt.plot(ols.coef_, color='yellowgreen', linestyle=':', linewidth=2,
             label="OLS estimate")
    plt.plot(y, color='orange', linestyle='-', linewidth=2, label="Ground truth")
    plt.xlabel("Features")
    plt.ylabel("Values of the weights")
    plt.legend(loc=1)

    plt.subplot(3,1,2)
    # plt.figure(figsize=(6, 5))
    # plt.title("Histogram of the weights")
    plt.hist(clf.coef_, bins=len(X), color='navy', log=True)
    # plt.scatter(clf.coef_[relevant_features], 5 * np.ones(len(relevant_features)),
    #             color='gold', marker='o', label="Relevant features")
    plt.ylabel("Features")
    # plt.xlabel("Values of the weights")
    plt.legend(loc=1)

    plt.subplot(3,1,3)
    # plt.figure(figsize=(6, 5))
    # plt.title("Marginal log-likelihood")
    plt.plot(clf.scores_, color='navy', linewidth=2)
    plt.ylabel("Score")
    # plt.xlabel("Iterations")

    plt.show()

# BRRegression(X,y)

#Ridge Regression

def RRegression(X,y,title):
    # clf = BayesianRidge(compute_score=True)
    # clf.fit(X, y)
    # clf_pred=clf.predict(X)

    n_alphas = 1000
    alphas = np.logspace(-5, 3, n_alphas)
    coefs = []
    for a in alphas:
        ridge = Ridge(alpha=a, fit_intercept=False)
        ridge.fit(X, y)
        coefs.append(ridge.coef_)

    reg_cv = RidgeCV(alphas)
    reg_cv.fit(X, y)

    reg = Ridge(alpha=reg_cv.alpha_)
    # reg = Ridge(alpha=100)
    reg.fit(X,y)
    reg_pred = reg.predict(X)

    plt.subplots_adjust(bottom=.12,hspace=.55,top=.93)
    plt.subplot(2,1,1)
    # plt.plot(X.WordMentions, y, 'r.', markersize=3)
    # plt.plot(X.WordMentions, reg_pred, 'b-')
    plt.plot(dt,y,'r.',markersize=3)
    plt.plot(dt,reg_pred,'b-')
    plt.gcf().autofmt_xdate()

    plt.ylabel(title)
    # plt.plot(m,y,'g-')



    plt.subplot(2,1,2)
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.axis('tight')


    print('The coefficient is: ')
    print(reg.coef_)
    print('The w0 is %.2f' %reg.intercept_)
    print('The most robust alpha is %.2f' %reg_cv.alpha_)
    print('RMSE: %.4f'% np.sqrt(np.sum((reg_pred-y)**2)/len(y)))
    plt.show()



RRegression(X,y,'Return')
# RRegression(X,ts_log_time,'Close')
# print(X)











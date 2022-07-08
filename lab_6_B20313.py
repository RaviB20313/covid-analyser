"""
Name = Ravi Ranjan
Roll no. = B20313
lab=6
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg as AR
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr

#Reading CSV
data=pd.read_csv("daily_covid_cases.csv")
#--------1(a)--------
data["Date"] = pd.to_datetime(data["Date"]) 
dates = data["Date"]
new_cases = data["new_cases"]
plt.plot(dates, new_cases, label="new_cases")
plt.xlabel("Date")
plt.ylabel("new_cases")
plt.xticks(rotation=45)
plt.show()

#---------1(b)---------
def getLagged(n_Days,data):
    return data[n_Days:].values[:,1]
def getAutoCorrelation(n_Days,originaldata):
    laggeddata=getLagged(n_Days,originaldata)
    data=originaldata.values[:,1]
    data=data[0:(data.size-n_Days)]
    assert(data.size==laggeddata.size)
    return np.corrcoef(data.astype("float32"), laggeddata.astype("float32"))[0,1]

print("------1(b)-------")
laggeddata1=getLagged(1,data)
print("Pearson_correlation_coefficient_between_the_generated_one_day_lag_time_sequence_and_the_given_time_sequence:",getAutoCorrelation(1,data))
print()


#------1(c)------
originaldata1=(data.values[:,1])
originaldata1=originaldata1[0:(originaldata1.size-1)]
plt.scatter(laggeddata1,originaldata1)
plt.title("original_Vs_lagged data")
plt.xlabel("Original_data")
plt.ylabel("Lagged_data")
plt.show()
print()

print("-------1(d)-------")
x=[]
y=[]
for i in range(1,7):
    x.append(i)
    y.append(getAutoCorrelation(i,data))
print(y)
plt.plot(x,y)
plt.title("Auto_Correlation vs Lagged_Value")
plt.ylabel("Auto_Correlation")
plt.xlabel("Lagged_Value")
plt.show()
print()

#--------1(e)--------
plot_acf(data.values[:,1],lags=list(range(1,8)))
plt.title("Auto_Correlation vs Lagged_Value")
plt.ylabel("Auto_Correlation")
plt.xlabel("Lagged_Value")
plt.show()
print()

#---------2(a)----------
print("-----2-----")
data = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')

test_size = 0.35 
X = data.values
tst_sz = int(np.ceil(len(X)*test_size))
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]
date_test=[data.iloc[i][0] for i in range(397,612)]
date_train=[data.iloc[i][0] for i in range(0,399)]
month_yr_train=["Feb-20","Apr-20","Jun-20","Aug-20","Oct-20","Dec-20","Feb-21"]
month_yr_test=["Feb-21","Apr-21","Jun21","Aug-21","Oct-21"]
plt.plot(train)
plt.xticks(np.arange(1,397,57),month_yr_train,rotation=45)
plt.title("First_wave_train")
plt.xlabel("month_year")
plt.ylabel("confirmed_cases")
plt.show()
plt.plot(test)
plt.title("Second_wave_test")
plt.xticks(np.arange(1,214,43),month_yr_test,rotation=45)
plt.xlabel("month_year")
plt.ylabel("confirmed_cases")
plt.show()


def predict(window):
    model = AR(train, lags=window) 
    model_fit = model.fit()
    coef = model_fit.params 
    if(window==5):
        print("coefficients:",coef)
    #-----2(b)--------
    hist = train[len(train)-window:]
    hist = [hist[i] for i in range(len(hist))]
    predic = list()
    for t in range(len(test)):
        length = len(hist)
        lag = [hist[i] for i in range(length-window,length)]
        y = coef[0] 
        for d in range(window):
            y += coef[d+1] * lag[window-d-1] 
        obs = test[t]
        predic.append(y) 
        hist.append(obs) 
    return predic
def mape(predic):
    return np.sum(abs(np.array(test)-np.array(predic))/(np.array(test)))*100/215
def rmse(predic):
    return np.sqrt(mse(test,predic))*100/(sum(test)/215)[0]
predic=predict(5)
plt.scatter(test,predic)
plt.xticks(rotation=45)
plt.title("original_vs_predicted_values")
plt.xlabel("original_confirmed_cases")
plt.ylabel("predicted_confirmed_cases")
plt.show()
#--------2(ii)----------
plt.plot(test)
plt.plot(predic)
plt.xticks(np.arange(1,214,43),month_yr_test,rotation=45)
plt.title("original_vs_predicted values")
plt.xlabel("month_year")
plt.ylabel("confirmed_cases")
plt.legend(["original","predicted"])
plt.show()
print("------2(iii)------")
print("RMSE%= ",rmse(predic),"%")
print("MAPE%= ",mape(predic),"%")

#--------3-----------
pre1=predict(1)
pre5=predic
pre10=predict(10)
pre15=predict(15)
pre25=predict(25)
l=['1','5','10','15','25']
rmse_lag=[rmse(pre1),rmse(pre5),rmse(pre10),rmse(pre15),rmse(pre25)]
mape_lag=[mape(pre1),mape(pre5),mape(pre10),mape(pre15),mape(pre25)]
plt.bar(l,rmse_lag)
plt.xlabel("lag_p")
plt.ylabel("rmse%")
plt.show()
plt.bar(l,mape_lag)
plt.xlabel("lag_p")
plt.ylabel("mape%")
plt.show()

print("-------4-------")
p = 1
while p < len(data):
  corr = pearsonr(train[p:].ravel(), train[:len(train)-p].ravel())
  if(abs(corr[0]) <= 2/math.sqrt(len(train[p:]))):
    print('heuristic_value_for_optimal_lags:',p-1)
    break
  p+=1

p=p-1
model = AR(train, lags=p)
model_fit = model.fit()
coef = model_fit.params 
hist = train[len(train)-p:]
hist = [hist[i] for i in range(len(hist))]
predicted = list() 
for t in range(len(test)):
  length = len(hist)
  Lag = [hist[i] for i in range(length-p,length)] 
  yhat = coef[0] 
  for d in range(p):
    yhat += coef[d+1] * Lag[p-d-1] 
  ob = test[t]
  predicted.append(yhat) 
  hist.append(ob) 

mape = np.mean(np.abs((test - predicted)/test))*100
rmse_per = (math.sqrt(mse(test, predicted))/np.mean(test))*100
print('RMSE%:',rmse_per)
print('MAPE%:',mape)

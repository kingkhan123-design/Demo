import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
dataset = pd.read_csv("C:/Users/mathu/Downloads/perrin-freres-monthly-champagne-.csv")
print(dataset.head())
print(dataset.info())
dataset.columns=["Month","Sales"]
print(dataset.head())
print(dataset.tail())
dataset.drop(106,axis=0,inplace=True)
dataset.drop(105,axis=0,inplace=True)
print(dataset.info())
#converting month coilum from obj to datetime
dataset["Month"]=pd.to_datetime(dataset["Month"])
print(dataset.info())
dataset.set_index("Month",inplace=True)
print(dataset.head())
dataset.plot()
plt.show()

#moving avg
#SMA over a period of 2 and 12 months
#min_period=min value to statrt calculation

dataset['SMA_2'] = dataset.Sales.rolling(2, min_periods=1).mean()
dataset['SMA_12'] = dataset.Sales.rolling(12, min_periods=1).mean()
print(dataset.head(20))
dataset.plot()
plt.show()

#SIMPLE EXPONENIAL MOVING AVG
# EMA Sales
#Exponential Moving Average (EMA) does a superb job in capturing the pattern of the data (0,1)
# Let's smoothing factor - 0.1
dataset['EMA_0.1'] = dataset.Sales.ewm(alpha=0.1, adjust=False).mean()
# Let's smoothing factor  - 0.3 value of alpha lie between 0 to 1
dataset['EMA_0.3'] = dataset.Sales.ewm(alpha=0.3, adjust=False).mean()
dataset.plot()
plt.show()

#Checking for the stationary data
rolmean = dataset.Sales.rolling(window=12).mean()
rolstd = dataset.Sales.rolling(window=12).std()
orig = plt.plot(dataset.Sales,color='blue',label='original')
mean = plt.plot(rolmean,color='red',label='Rolling Mean')
std = plt.plot(rolstd,color='black',label='Rolling std')
plt.legend()
plt.title('Rolling mean and Std deviation')
plt.show()

#testing for the stationary using Dickey fuller test
from statsmodels.tsa.stattools import adfuller
test_result=adfuller(dataset["Sales"])
print(test_result)

#Alternative of above
def adfuller_test(sales):
    result=adfuller(sales)
    #print(result)
    labels = ['ADF Test Statistic','p-value','Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
    else:
        print("weak evidence against null hypothesis, indicating it is non-stationary ")

adfuller_test(dataset["Sales"])#p-value : 0.3639157716602467 should be less than 0.05 // this data is not stationary

#Converting Non stationary to stationary
# Detrending
dataset_detrend =  (dataset['Sales'] - dataset['Sales'].rolling(window=12).mean())/dataset['Sales'].rolling(window=12).std()
rolmean = dataset_detrend.rolling(window=12).mean()
rolstd = dataset_detrend.rolling(window=12).std()
orig = plt.plot(dataset_detrend,color='blue',label='original')
mean = plt.plot(rolmean,color='red',label='Rolling Mean')
std = plt.plot(rolstd,color='black',label='Rolling std')
plt.legend()
plt.title('Rolling mean and Std deviation')
plt.show()
#DIFFEENCES
dataset['Seasonal_Difference']=dataset['Sales']-dataset['Sales'].shift(12)
print(dataset.head(20))
adfuller_test(dataset["Seasonal_Difference"].dropna())

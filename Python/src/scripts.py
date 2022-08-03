# -*- coding: utf-8 -*-
# Import general libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import statistics
import ulmo
from datetime import datetime

# importing relevant library
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf

#Set global parameters
## Today's date
today = datetime.today().strftime('%Y-%m-%d')

## Set API url
wsdlurl = 'https://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL'


#########################
#Begin defined functions#
#########################

def Multi_ARIMA(dictionary, order=(0,0,0), seasonal_order=(0,0,0,0), plot = False):
    # performs an ARIMA model on each column of the dataset 
    # and returns the mean RMSE from a walk-forward validation
    from statistics import mean
    rmse_list = []
    for i in dictionary.keys():
        inp = pd.DataFrame(dictionary[i])
        
        #formatting the dataframe
        inp.fillna(method='ffill', inplace=True)
        inp['Date'] = inp.index
        inp.index = pd.to_datetime(inp['Date']).apply(lambda x: x.date())
        inp.drop(columns=['Date'], inplace=True)
        
        cutoff = round(inp.shape[0]*0.8)
        x_train = inp[:cutoff]
        x_test = inp[cutoff:]
        
        predictions = []
        history = x_train
        
        # walk-forward validation
        for t in range(len(x_test)):
            model = ARIMA(history, order=order, seasonal_order=seasonal_order)
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = x_test['value'][t]
            history.append(pd.DataFrame(index=[obs]))
        #   print('predicted=%f, expected=%f' % (yhat, obs))
            
        if plot == True:
            # plot forecasts against actual outcomes
            x_predictions = pd.DataFrame(predictions)
            x_predictions = x_predictions.set_index(x_test.index)
            plt.plot(x_train, color='blue')
            plt.plot(x_test, color='blue')
            plt.plot(x_predictions, color='red')
            plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
            plt.title(f'Time Series for {i}')
            plt.legend(loc='upper left', fontsize=8)
            plt.show()
            print(f'RMSE: {mean_squared_error(x_test, predictions, squared=False)}')
    
        # evaluate forecasts
        rmse_list.append(mean_squared_error(x_test, predictions, squared=False))
    return mean(rmse_list)



# Define data fetch function
def snotel_fetch(sitecode, variablecode, start_date, end_date=today):
    values_df = None
    try:
        # Request data from the server
        site_values = ulmo.cuahsi.wof.get_values(wsdlurl, sitecode, variablecode, start=start_date, end=end_date)
        # Convert to a Pandas DataFrame   
        values_df = pd.DataFrame.from_dict(site_values['values'])
        # Parse the datetime values to Pandas Timestamp objects
        values_df['datetime'] = pd.to_datetime(values_df['datetime'], utc=True)
        # Set the DataFrame index to the Timestamps
        values_df = values_df.set_index('datetime')
        # Convert values to float and replace -9999 nodata values with NaN
        values_df['value'] = pd.to_numeric(values_df['value']).replace(-9999, np.nan)
        # Remove any records flagged with lower quality
        values_df = values_df[values_df['quality_control_level_code'] == '1']
    except:
        print("Unable to fetch %s" % variablecode)

    return values_df



# Stationarity function by Jacob Stallone 
# (https://medium.com/@stallonejacob/time-series-forecast-a-basic-introduction-using-python-414fcb963000)
# Updated here.
def test_stationarity(timeseries, period):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(period).mean()
    rolstd = timeseries.rolling(period).std()
    
    #Plot rolling statistics:
    timeseries.plot( color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

    
    
    
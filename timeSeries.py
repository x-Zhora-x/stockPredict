import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from subprocess import check_output
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pandas.plotting import lag_plot
import pmdarima as pm

from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

aliba = pd.read_csv("AABA_2006-01-01_to_2018-01-01.csv")
apple = pd.read_csv("AAPL_2006-01-01_to_2018-01-01.csv")
amazon = pd.read_csv("AMZN_2006-01-01_to_2018-01-01.csv")
cisco = pd.read_csv("CSCO_2006-01-01_to_2018-01-01.csv")
google = pd.read_csv("GOOGL_2006-01-01_to_2018-01-01.csv")
ibm = pd.read_csv("IBM_2006-01-01_to_2018-01-01.csv")

ibm = ibm.rename(columns={"Close" : "Ibm"})

ibm = ibm.rename(columns={"Close" : "Ibm"})
apple = apple.rename(columns={"Close" : "Apple"})
amazon = amazon.rename(columns={"Close" : "Amazon"})
cisco = cisco.rename(columns={"Close" : "Cisco"})
google = google.rename(columns={"Close" : "Google"})
aliba = aliba.rename(columns={"Close" : "Aliba"})





ibm = ibm.drop(["Open", "High","Low","Volume","Name"] , axis = 1)
apple = apple.drop(["Open", "High","Low","Volume","Name"] , axis = 1)
amazon = amazon.drop(["Open", "High","Low","Volume","Name"] , axis = 1)
cisco = cisco.drop(["Open", "High","Low","Volume","Name"] , axis = 1)
google = google.drop(["Open", "High","Low","Volume","Name"] , axis = 1)
aliba = aliba.drop(["Open", "High","Low","Volume","Name"] , axis = 1)


ibm = ibm.set_index("Date")
ibm.index = pd.to_datetime(ibm.index)
adf_results = adfuller(ibm["Ibm"])





adf_results_2 = adfuller(ibm["Ibm"].diff().dropna())
#print(adf_results)
#print(adf_results_2)

#plot_acf(ibm["Ibm"])

#plot_pacf(ibm["Ibm"])

import statsmodels.tsa.stattools as ts
def dftest(timeseries):
    dftest = ts.adfuller(timeseries,)
    dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic','p­value','Lags Used','Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)    
    rolmean = timeseries.rolling(window = 12).mean()
    rolstd = timeseries.rolling(window = 12).std()

    orig = plt.plot(timeseries, color = "blue", label = "Original" )
    mean = plt.plot(rolmean, color = "red", label = "Rolling Mean" )
    plt.legend(loc  = "best")
    plt.title("Rolling  Mean  and  Standard  Deviation")
    plt.grid()
    plt.show()
    
annual = ibm.resample('A').mean()

'''
#sns.violinplot(x = ibm.index.month, y = ibm["Ibm"])

from statsmodels.tsa.seasonal import seasonal_decompose
#seasonal_decompose(ibm.Ibm, period = 12).plot()

#results = pm.auto_arima(ibm, seasonal = True, m = 12, trace = True)



train = ibm.iloc[:int(len(ibm)*0.95)]

test = ibm.iloc[int(len(ibm)*0.95):]



model = SARIMAX(test)

sarima_results = model.fit()

sarima_pred = sarima_results.get_forecast(steps=len(test))
sarima_mean = sarima_pred.predicted_mean

plt.plot(test.index, sarima_mean, label='SARIMA')
plt.plot(test.index, test, label='observed')
plt.legend()
#plt.show()
'''

dataset = ibm.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


train_data = scaled_data[0:int(training_data_len), :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_train.shape


from keras.models import Sequential
from keras.layers import Dense, LSTM

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()













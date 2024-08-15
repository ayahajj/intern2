#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


gstock_data = pd.read_csv('C:/Users/USER/Downloads/tesla-stock-price.csv')
gstock_data .head()


# In[3]:


# Select specific columns
gstock_data = gstock_data[['date', 'open', 'close']]

# Convert the 'date' column to datetime, keeping only the date part
gstock_data['date'] = pd.to_datetime(gstock_data['date'].apply(lambda x: x.split()[0]))

# Set 'date' as the index and drop the original 'date' column
gstock_data.set_index('date', drop=True, inplace=True)

# Display the first few rows of the DataFrame
gstock_data.head()


# In[4]:


import matplotlib.pyplot as plt

# Create subplots with 1 row and 2 columns, and a specified figure size
fg, ax = plt.subplots(1, 2, figsize=(20, 7))

# Plot 'open' prices on the first subplot
ax[0].plot(gstock_data['open'], label='Open', color='green')
ax[0].set_xlabel('Date', size=15)
ax[0].set_ylabel('Price', size=15)
ax[0].legend()

# Plot 'close' prices on the second subplot
ax[1].plot(gstock_data['close'], label='Close', color='red')
ax[1].set_xlabel('Date', size=15)
ax[1].set_ylabel('Price', size=15)
ax[1].legend()

# Display the plots
fg.show()


# In[5]:


from sklearn.preprocessing import MinMaxScaler
Ms = MinMaxScaler()
gstock_data [gstock_data .columns] = Ms.fit_transform(gstock_data )

training_size = round(len(gstock_data ) * 0.80)

train_data = gstock_data [:training_size]
test_data  = gstock_data [training_size:]


# In[9]:


def create_sequence(dataset):
    sequences = []
    labels = []

    start_idx = 0

    for stop_idx in range(50, len(dataset)): 
        sequences.append(dataset.iloc[start_idx:stop_idx])
        labels.append(dataset.iloc[stop_idx])
        start_idx += 1

    return np.array(sequences), np.array(labels)

# Assuming train_data and test_data are already defined
train_seq, train_label = create_sequence(train_data)
test_seq, test_label = create_sequence(test_data)


# In[8]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional


# In[13]:


def create_sequence(dataset):
    sequences = []
    labels = []

    start_idx = 0

    for stop_idx in range(50, len(dataset)):
        sequences.append(dataset.iloc[start_idx:stop_idx])
        labels.append(dataset.iloc[stop_idx])
        start_idx += 1
    return (np.array(sequences), np.array(labels))

# Assuming `train_data` is your dataset
train_seq, train_label = create_sequence(train_data)


# In[14]:


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape = (train_seq.shape[1], train_seq.shape[2])))

model.add(Dropout(0.1)) 
model.add(LSTM(units=50))

model.add(Dense(2))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

model.summary()


# In[20]:


test_seq, test_label = create_sequence(test_data)
model.fit(train_seq, train_label, epochs=80,validation_data=(test_seq, test_label), verbose=1)
test_predicted = model.predict(test_seq)
test_inverse_predicted = MMS.inverse_transform(test_predicted)


# In[22]:


from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
MMS = MinMaxScaler()

# Fit the scaler on your training data (assuming you're scaling 'open' and 'close' columns)
MMS.fit(gstock_data[['open', 'close']])

# Scale your training and test data
train_scaled = MMS.transform(train_data[['open', 'close']])
test_scaled = MMS.transform(test_data[['open', 'close']])

# After training the model and making predictions
test_predicted = model.predict(test_seq)


# In[27]:


# Adjusting gstock_data slicing to match test_inverse_predicted length
gs_slic_data = pd.concat([
    gstock_data.iloc[-101:].copy(),
    pd.DataFrame(test_inverse_predicted, columns=['open_predicted', 'close_predicted'], index=gstock_data.iloc[-101:].index)
], axis=1)
test_seq, test_label = create_sequence(gstock_data.iloc[-202:])
gs_slic_data[['open','close']] = MMS.inverse_transform(gs_slic_data[['open','close']])


# In[29]:


import matplotlib.pyplot as plt

# Plotting the actual vs predicted open price
gs_slic_data[['open','open_predicted']].plot(figsize=(10,6))
plt.xticks(rotation=45)
plt.xlabel('Date', size=15)
plt.ylabel('Stock Price', size=15)
plt.title('Actual vs Predicted for open price', size=15)
plt.show()


# In[31]:


import matplotlib.pyplot as plt

# Plotting the actual vs predicted close price
gs_slic_data[['close', 'close_predicted']].plot(figsize=(10,6))
plt.xticks(rotation=45)
plt.xlabel('Date', size=15)
plt.ylabel('Stock Price', size=15)
plt.title('Actual vs Predicted for close price', size=15)
plt.show()


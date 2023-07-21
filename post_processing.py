import pickle as pkl
import keras
import pandas as pd
import tensorflow as tf
import os
from data_formatters import ohlc
from libs import utils,tft_model

import pandas as pd
import matplotlib.pyplot as plt



# load dataset
df = pd.read_csv(os.path.join('output','data','ohlc','ohlc.csv'))
df['Datetime'] = pd.to_datetime(df['Datetime'])
min_date = df['Datetime'].agg(['min'])[0]
max_date = df['Datetime'].agg(['max'])[0]
valid_boundary = min_date+(max_date-min_date)/2
test_boundary = valid_boundary+((max_date-min_date)/4)

index = df['Datetime']
train = df.loc[index < valid_boundary]
valid = df.loc[(index >= valid_boundary) & (index < test_boundary)]
test = df.loc[(index >= test_boundary)] #& (df.index <= '2019-06-28')]
print('Formatting train-valid-test splits.')

#load the model
model_folder = os.path.join('output', 'saved_models', 'ohlc', 'fixed')
session = tf.compat.v1.keras.backend.get_session()
model = tft_model.TemporalFusionTransformer()
checkpoint_path = os.path.join('output', 'saved_models', 'ohlc', 'fixed', 'checkpoint')
model.load_weights(checkpoint_path)
#model = tf.keras.models.load_model(filepath=model_folder, compile=False)#pkl.load('model.pickle')#utils.load(tf_session=tf.compat.v1.keras.backend.get_session(),model_folder=model_folder, cp_name='TemporalFusionTransformer', scope='TemporalFusionTransformer')
#trained_model = keras.models.load_model("trained_model.keras")

for i in range(0,len(test)):
    row = test.iloc[i]
    pred = model.predict(row)
    #if prediction mean is higher than t0, sell, if lower, buy

file = open('output_mape.pickle', 'rb')
# mean capital gain, number of trades
# 1. curves capital, stock substract capital from market curve, then average
# 2. number of trades (average number of trades per day) all curves as percentage difference to first day 0
data = pkl.load(file)
targets = data['targets']
mean = targets[['t+0', 't+1', 't+2', 't+3', 't+4']].mean(axis=1)
# close the file
plt.plot(targets['t+0'][:100], color='red')
plt.plot(mean[:100], color='blue')
plt.show()
file.close()
x = [0,1,2,3,4]
for i in range(0, 50):
    row = targets.iloc[i]
    y = [row['t+0'], row['t+1'], row['t+2'], row['t+3'], row['t+4']]
    plt.plot(x, y)
    x = [i+1 for i in x]
plt.show()

# do prediction per ticker and ohlc value
# for each prediction: if value above initial value: buy; else sell (do nothing if not invested)
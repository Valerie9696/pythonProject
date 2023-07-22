import json
import pickle as pkl

import dill
import keras
import pandas as pd
import tensorflow.compat.v1 as tf
import os
from data_formatters import ohlc
from libs import utils,tft_model
import expt_settings.configs
import libs.hyperparam_opt
import joblib
import libs

import pandas as pd
import matplotlib.pyplot as plt


# load dataset
ExperimentConfig = expt_settings.configs.ExperimentConfig
config = ExperimentConfig('ohlc', 'output')
data_formatter = config.make_data_formatter()
data_csv_path = os.path.join('output','data','ohlc','ohlc.csv')
raw_data = pd.read_csv(data_csv_path, index_col=0)
train, valid, test = data_formatter.split_data(raw_data)


#load the model
model_folder = os.path.join('output', 'saved_models', 'ohlc', 'fixed')
use_gpu=False
name = 'ohlc'
output_folder = 'output'
ExperimentConfig = expt_settings.configs.ExperimentConfig
config = ExperimentConfig(name, output_folder)
data_formatter = config.make_data_formatter()
fixed_params = data_formatter.get_experiment_params()
params = data_formatter.get_default_model_params()
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
opt_manager = HyperparamOptManager({k: [params[k]] for k in params},
                                       fixed_params, model_folder)
ModelClass = libs.tft_model.TemporalFusionTransformer
if use_gpu:
    tf_config = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=0)

else:
    tf_config = utils.get_default_tensorflow_config(tf_device="cpu")
with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
    tf.keras.backend.set_session(sess)
    #model = utils.load(tf_session=sess, model_folder='trained_model', cp_name='TemporalFusionTransformer')

a = dill.loads(
    b"\x80\x04\x95P\x04\x00\x00\x00\x00\x00\x00XI\x04\x00\x00[10, 5, 20]_[('Symbol', <DataTypes.CATEGORICAL: 1>, <InputTypes.ID: 4>), ('Datetime', <DataTypes.DATE: 2>, <InputTypes.TIME: 5>), ('Open', <DataTypes.REAL_VALUED: 0>, <InputTypes.OBSERVED_INPUT: 1>), ('High', <DataTypes.REAL_VALUED: 0>, <InputTypes.OBSERVED_INPUT: 1>), ('Low', <DataTypes.REAL_VALUED: 0>, <InputTypes.OBSERVED_INPUT: 1>), ('HighLowDifference', <DataTypes.REAL_VALUED: 0>, <InputTypes.OBSERVED_INPUT: 1>), ('OpenCloseDifference', <DataTypes.REAL_VALUED: 0>, <InputTypes.OBSERVED_INPUT: 1>), ('ATR', <DataTypes.REAL_VALUED: 0>, <InputTypes.OBSERVED_INPUT: 1>), ('RSI', <DataTypes.REAL_VALUED: 0>, <InputTypes.OBSERVED_INPUT: 1>), ('Close', <DataTypes.REAL_VALUED: 0>, <InputTypes.TARGET: 0>), ('HoursFromStart', <DataTypes.REAL_VALUED: 0>, <InputTypes.KNOWN_INPUT: 2>), ('HourOfDay', <DataTypes.CATEGORICAL: 1>, <InputTypes.KNOWN_INPUT: 2>), ('DayOfWeek', <DataTypes.CATEGORICAL: 1>, <InputTypes.KNOWN_INPUT: 2>), ('StatSymbol', <DataTypes.CATEGORICAL: 1>, <InputTypes.STATIC_INPUT: 3>)]_0.1_5_5_[7]_12_[0, 1, 2]_[8]_0.01_1.0_64_output\\saved_models\\ohlc\\fixed_5_252_1_1_1_1_[11]_257\x94.")
with open('optimal_name.json', 'rb') as f:
    j = json.load(f)
    f.close()
opt_manager.optimal_name = a  # dill.loads(b'\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x94.')
opt_manager.hyperparam_folder = os.path.join('output','saved_models','ohlc','fixed')
opt_manager._override_w_fixed_params = True
with open('fixed_params.json', 'rb') as b:
    #j = json.load(b)
    opt_manager.fixed_params = json.load(b)
    b.close()

with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
    tf.keras.backend.set_session(sess)
    best_params = opt_manager.get_best_params()
    #with open('optimal_name.txt', 'r') as f:
     #   opt_manager.optimal_name = dill.load(f)
    sess.run(tf.global_variables_initializer())
    model = ModelClass(raw_params=best_params, use_cudnn=use_gpu)
    #model.model.load_weights(filepath='trained_model')
    model.load(opt_manager.hyperparam_folder)
    #for i in range(0, len(test)):
        #row = test.iloc[i]
    pred = model.predict(test, return_targets=True)
    a=0
#checkpoint_path = os.path.join('output', 'saved_models', 'ohlc', 'fixed', 'checkpoint')
#model = joblib.load('model.pkl')
#model.load_weights(checkpoint_path)
#model = tf.keras.models.load_model(filepath=model_folder, compile=False)#pkl.load('model.pickle')#utils.load(tf_session=tf.compat.v1.keras.backend.get_session(),model_folder=model_folder, cp_name='TemporalFusionTransformer', scope='TemporalFusionTransformer')
#trained_model = keras.models.load_model("trained_model")


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
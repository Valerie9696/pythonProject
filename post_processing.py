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
import libs
import pandas as pd
import matplotlib.pyplot as plt

def get_test_data(df):
    #df['Datetime'] = pd.to_datetime(df['Datetime'])
    min_date = df['Datetime'].agg(['min'])[0]
    max_date = df['Datetime'].agg(['max'])[0]
    valid_boundary = min_date + (max_date - min_date) / 2
    test_boundary = valid_boundary + ((max_date - min_date) / 4)

    index = df['Datetime']
    train = df.loc[index < valid_boundary]
    valid = df.loc[(index >= valid_boundary) & (index < test_boundary)]
    test = df.loc[(index >= test_boundary)]
    return test
def prep_data(df, data_formatter):
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df_transformed = data_formatter.transform_inputs(df=df)
    return df_transformed
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

with open('optimal_name.json', 'rb') as f:
    j = json.load(f)
    f.close()
opt_manager.optimal_name = dill.loads(b"\x80\x04\x95P\x04\x00\x00\x00\x00\x00\x00XI\x04\x00\x00[10, 5, 20]_[('Symbol', <DataTypes.CATEGORICAL: 1>, <InputTypes.ID: 4>), ('Datetime', <DataTypes.DATE: 2>, <InputTypes.TIME: 5>), ('Open', <DataTypes.REAL_VALUED: 0>, <InputTypes.OBSERVED_INPUT: 1>), ('High', <DataTypes.REAL_VALUED: 0>, <InputTypes.OBSERVED_INPUT: 1>), ('Low', <DataTypes.REAL_VALUED: 0>, <InputTypes.OBSERVED_INPUT: 1>), ('HighLowDifference', <DataTypes.REAL_VALUED: 0>, <InputTypes.OBSERVED_INPUT: 1>), ('OpenCloseDifference', <DataTypes.REAL_VALUED: 0>, <InputTypes.OBSERVED_INPUT: 1>), ('ATR', <DataTypes.REAL_VALUED: 0>, <InputTypes.OBSERVED_INPUT: 1>), ('RSI', <DataTypes.REAL_VALUED: 0>, <InputTypes.OBSERVED_INPUT: 1>), ('Close', <DataTypes.REAL_VALUED: 0>, <InputTypes.TARGET: 0>), ('HoursFromStart', <DataTypes.REAL_VALUED: 0>, <InputTypes.KNOWN_INPUT: 2>), ('HourOfDay', <DataTypes.CATEGORICAL: 1>, <InputTypes.KNOWN_INPUT: 2>), ('DayOfWeek', <DataTypes.CATEGORICAL: 1>, <InputTypes.KNOWN_INPUT: 2>), ('StatSymbol', <DataTypes.CATEGORICAL: 1>, <InputTypes.STATIC_INPUT: 3>)]_0.1_5_5_[7]_12_[0, 1, 2]_[8]_0.01_1.0_64_output\\saved_models\\ohlc\\fixed_5_252_1_1_1_1_[11]_257\x94.")
opt_manager.hyperparam_folder = os.path.join('output','saved_models','ohlc','fixed')
opt_manager._override_w_fixed_params = True
with open('fixed_params.json', 'rb') as b:
    opt_manager.fixed_params = json.load(b)
    b.close()

budget = 100000

with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
    tf.keras.backend.set_session(sess)
    best_params = opt_manager.get_best_params()
    sess.run(tf.global_variables_initializer())
    model = ModelClass(raw_params=best_params, use_cudnn=use_gpu)
    model.load(opt_manager.hyperparam_folder)
    raw_test = get_test_data(raw_data)
    budget_dict = {col: [] for col in data_formatter.identifiers}
    id = '.1COV'
    #budget_frame = df = pd.DataFrame(columns=data_formatter.identifiers)
    for i in range(257, len(raw_test)-1):
        df = raw_test.iloc[i-257:i]
        raw_row = raw_test.iloc[[i]]
        buy_value = raw_row['Close'].iloc[0]
        cur_id = raw_row['Symbol'].iloc[0]
        if df['Symbol'].nunique() > 1:
            id = cur_id
            if i+257 < len(raw_test)-1:
                i = i + 257
            else:
                break
        else:
            cur_time = raw_row['Datetime'].iloc[0]
            df_transformed = data_formatter.transform_inputs(df=df)
            scaled_row = df_transformed.iloc[[256]]
            cur_value = scaled_row['Close'].iloc[0]
            pred = model.predict(df_transformed, return_targets=True)
            targets = pred['targets']
            mean = targets[['t+0', 't+1', 't+2', 't+3', 't+4']].mean(axis=1).iloc[0]
            # buy if mean of future values is smaller than current value
            if mean < cur_value-1:
                price = 10 * buy_value
                budget = budget - price
                triplet = (cur_time, price, budget)
                budget_dict[cur_id].append(triplet)
            #sell if mean of future values is bigger than current value
            elif mean > cur_value+1:
                price = 10*buy_value
                budget = budget+price
                triplet = (cur_time, price, budget)
                budget_dict[cur_id].append(triplet)
    dict_items = budget_dict.items()
    counter = 0
    for dict in dict_items:
        budget_df = pd.DataFrame.from_records(dict[1], columns =['Datetime', 'Price', 'Budget'])#pd.DataFrame.from_dict(dict)
        path = os.path.join('testing', 'results', data_formatter.identifiers[counter]+'.csv')
        budget_df.to_csv(path)
        counter = counter+1

    #if prediction mean is higher than t0, sell, if lower, buy

#file = open('output_mape.pickle', 'rb')
# mean capital gain, number of trades
# 1. curves capital, stock substract capital from market curve, then average
# 2. number of trades (average number of trades per day) all curves as percentage difference to first day 0

# do prediction per ticker and ohlc value
# for each prediction: if value above initial value: buy; else sell (do nothing if not invested)
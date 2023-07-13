import pickle as pkl

import pandas as pd

file = open('output_mape.pickle', 'rb')

# dump information to that file
data = pkl.load(file)
targets = data['targets']
mean = targets[['t+0', 't+1', 't+2', 't+3', 't+4']].mean(axis=1)
# close the file
file.close()

#load the model
# do prediction per ticker and ohlc value
# for each prediction: if value above initial value: buy; else sell (do nothing if not invested)
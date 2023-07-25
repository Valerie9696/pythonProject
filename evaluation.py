import os
import pandas as pd
import matplotlib.pyplot as plt

for file in os.listdir(os.path.join('testing', 'results')):
    df = pd.read_csv(os.path.join('testing', 'results',file))
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce', utc=True)
    df['date_only'] = df['Datetime'].dt.date
    # Group the DataFrame by 'date_only' and count the number of rows in each group
    count_per_day = df.groupby('date_only').size().reset_index(name='Count')
    mean_trades = count_per_day['Count'].mean()
    plt.title(file + ' budget')
    plt.plot(df['Budget'], color='blue', label='Budget')
    plt.legend()
    plt.show()
    plt.title(file+' trades per day')
    plt.plot(count_per_day['Count'], color='red', label='Trades per day')
    plt.legend()
    plt.show()
    print(count_per_day)
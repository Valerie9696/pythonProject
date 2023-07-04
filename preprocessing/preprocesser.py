import pandas as pd


def add_hour_column():
    df = pd.read_csv('ohlc.csv')
   # dt = df['Datetime']
    #print(dateinfer.infer([dt.iloc[1]]))
    #df['hour'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S%z').dt.hour.astype(float)
    hours = []
    for date in df['Datetime']:
        parts = date.split(' ')
        time = parts[1]
        hour = int(time[0:2])
        hours.append(hour)
    df['hour'] = hours
    df['filler_id'] = 'filler'
    df.reset_index(drop=True)
    df = df.iloc[:, 1:]
    df.to_csv('ohlc.csv')

add_hour_column()
('early stopping')
import pandas as pd
import mplfinance as mpf
import os

folder = "/home/trangndp/projects/trading_bot/create_dataset_rs_rl/csv/XAUUSD"
output_folder = "/home/trangndp/projects/trading_bot/create_dataset_rs_rl/image/XAUUSD"

for filename in os.listdir(folder):
    sub_ts_id = filename.split('.')[0].split('_')[-1]
    timeframe = filename.split('.')[0].split('_')[-2]

    file_path = os.path.join(folder, filename)
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    mpf.plot(df,
            type='candle',
            style='sas',
            volume=True,
            title=f'XAUUSDm {timeframe} - Candlestick Chart',
            ylabel='Price (USD)',
            ylabel_lower='Volume',
            figsize=(20, 8),
            figratio=(20, 8),
            figscale=1.2,
            tight_layout=True,
            xrotation=15,
            datetime_format='%b %d %Y %H:%M',
            savefig=f'{output_folder}/XAUUSDm_{timeframe}_{sub_ts_id}.png')


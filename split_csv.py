import os
import pandas as pd

output_folder = "/home/trangndp/projects/trading_bot/create_dataset_rs_rl/csv/XAUUSD"
os.makedirs(output_folder, exist_ok=True)

full_ts = pd.read_csv("/home/trangndp/projects/trading_bot/create_dataset_rs_rl/XAUUSD_M1.csv", index_col=0, sep='\t', header=0, parse_dates=True, encoding='utf-16')
ctx_len = 1440
sub_ts_id = 0

for i in range(0, full_ts.shape[0] - ctx_len + 1, ctx_len):
    sub_ts = full_ts.iloc[i:i+ctx_len]
    sub_ts_name = f"XAUUSDm_M1_{sub_ts_id}.csv"
    sub_ts_path = os.path.join(output_folder, sub_ts_name)
    sub_ts.to_csv(sub_ts_path, index=True)
    sub_ts_id += 1

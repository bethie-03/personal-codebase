import os
import random
import pandas as pd

import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from system_prompt import SYSTEM_PROMPT
from QWEN_TS.processing_qwen2_5_vl_ts import Qwen2_5_VLProcessor_TS

def get_percent_at_value(start_point, end_point, value_list):
    percent_list = []
    numbers = list(range(start_point, end_point + 1))  
    for value in value_list:
        index = numbers.index(value)
        percent = (index / (len(numbers) - 1)) * 100
        percent_list.append(percent)
    return percent_list

def get_value_at_percent(start_point, end_point, percent_list):
    value_list = []
    numbers = list(range(start_point, end_point + 1))  
    for percent in percent_list:
        if not 0 <= percent <= 100:
            raise ValueError("percent must be between 0 and 100")
        
        index = int((percent / 100) * (len(numbers) - 1))
        value_list.append(numbers[index])
    return value_list

def calculate_target(entry, pips, order_type, trade_type, pip_size=0.01):
    value = pips * pip_size
    if "buy" in order_type:
        if trade_type.lower() == "tp":
            return entry + value
        elif trade_type.lower() == "sl":
            return entry - value
        else:
            raise ValueError("trade_type must be 'tp' or 'sl'")
    elif "sell" in order_type:
        if trade_type.lower() == "tp":
            return entry - value
        elif trade_type.lower() == "sl":
            return entry + value
        else:
            raise ValueError("trade_type must be 'tp' or 'sl'")
    else:
        raise ValueError("trade_type must include 'buy' or 'sell'")
    
def calculate_lot_size(capital, risk_percent, stop_loss_pips, pip_value_per_lot=1.0):
    risk_amount = (risk_percent / 100) * capital
    lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
    return round(lot_size, 2)  

def calculate_profit_or_loss(lot_size, pip_diff, pip_value_per_lot=1.0):
    profit_or_loss = lot_size * pip_diff * pip_value_per_lot
    return round(profit_or_loss, 2)

def check_order_result(folder, df, sub_ts_id, period, entry_id, order_type, take_profit, stop_loss):
    df_check = df[entry_id:]
    is_win = None
    while is_win is None:
        if order_type == "buy":
            for i in range(df_check.shape[0]):
                if df_check.iloc[i]["High"] > take_profit:
                    is_win = True
                elif df_check.iloc[i]["Low"] < stop_loss:
                    is_win = False
        elif order_type == "sell":
            for i in range(df_check.shape[0]):
                if df_check.iloc[i]["High"] > stop_loss:
                    is_win = False
                elif df_check.iloc[i]["Low"] < take_profit:
                    is_win = True

        sub_ts_id += 1
        sub_filename = f"XAUUSDm_{period}_{sub_ts_id}.csv"
        sub_file_path = os.path.join(folder, sub_filename)
        try:
            df_next = pd.read_csv(sub_file_path, index_col=0)
        except:
            return None
        df_check = df_next[['Open', 'High', 'Low', 'Close', 'Volume']]

    return is_win

def check_instance_future_order_result(folder, sub_ts_id, period, order_type, take_profit, stop_loss):
    sub_ts_id += 1
    sub_filename = f"XAUUSDm_{period}_{sub_ts_id}.csv"
    sub_file_path = os.path.join(folder, sub_filename)
    try:
        df_next = pd.read_csv(sub_file_path, index_col=0)
    except:
        return None
    df_check = df_next[['Open', 'High', 'Low', 'Close', 'Volume']]

    is_win = None
    while is_win is None:
        if order_type == "buy":
            for i in range(df_check.shape[0]):
                if df_check.iloc[i]["High"] > take_profit:
                    is_win = True
                elif df_check.iloc[i]["Low"] < stop_loss:
                    is_win = False
        elif order_type == "sell":
            for i in range(df_check.shape[0]):
                if df_check.iloc[i]["High"] > stop_loss:
                    is_win = False
                elif df_check.iloc[i]["Low"] < take_profit:
                    is_win = True

    return is_win

def check_pending_future_order_result(folder, sub_ts_id, period, current_price, order_type, trigger_price, entry_price, take_profit, stop_loss):
    if "stop_limit" not in order_type:
        if trigger_price is not None:
            raise ValueError(f"{order_type} should only have one entry price, but got trigger={trigger_price} and limit={entry_price}")

    sub_ts_id += 1
    sub_filename = f"XAUUSDm_{period}_{sub_ts_id}.csv"
    sub_file_path = os.path.join(folder, sub_filename)

    try:
        df_next = pd.read_csv(sub_file_path, index_col=0)
    except:
        return None

    df_check = df_next[['Open', 'High', 'Low', 'Close', 'Volume']]

    triggered = False
    for i in range(df_check.shape[0]):
        high = df_check.iloc[i]["High"]
        low = df_check.iloc[i]["Low"]

        if order_type == "buy_limit" and low <= entry_price:
            triggered = True
            break
        elif order_type == "sell_limit" and high >= entry_price:
            triggered = True
            break
        elif order_type == "buy_stop" and high >= entry_price:
            triggered = True
            break
        elif order_type == "sell_stop" and low <= entry_price:
            triggered = True
            break
        elif order_type == "buy_stop_limit":
            if current_price < trigger_price:
                if high >= trigger_price and high >= entry_price:
                    triggered = True
                    break
            if current_price > trigger_price:
                if low <= trigger_price and high >= entry_price:
                    triggered = True
                    break
        elif order_type == "sell_stop_limit":
            if current_price < trigger_price:
                if high >= trigger_price and low <= entry_price:
                    triggered = True
                    break
            if current_price > trigger_price:
                if low <= trigger_price and low <= entry_price:
                    triggered = True
                    break
    if not triggered:
        return None  

    is_win = None
    for i in range(df_check.shape[0]):
        high = df_check.iloc[i]["High"]
        low = df_check.iloc[i]["Low"]

        if "buy" in order_type:
            if high >= take_profit:
                is_win = True
                break
            elif low <= stop_loss:
                is_win = False
                break
        elif "sell" in order_type:
            if low <= take_profit:
                is_win = True
                break
            elif high >= stop_loss:
                is_win = False
                break

    return is_win

def determine_action_for_pending_order(folder, order, order_id, df, current_price, entry_id, sub_ts_id, period, lookahead=2, min_rr=2.0):
    entry = order['entry']
    tp_pips = order['take_profit_pips']
    sl_pips = order['stop_loss_pips']
    order_type = order['order_type']

    rr = tp_pips / sl_pips
    if rr < min_rr:
        return [f"cancel {order_id}"]

    combined_df = df[entry_id:].copy()
    current_ts_id = sub_ts_id
    for _ in range(lookahead):
        current_ts_id += 1
        sub_filename = f"XAUUSDm_{period}_{current_ts_id}.csv"
        sub_file_path = os.path.join(folder, sub_filename)
        try:
            df_next = pd.read_csv(sub_file_path, index_col=0)[['Open', 'High', 'Low', 'Close', 'Volume']]
            combined_df = pd.concat([combined_df, df_next])
        except:
            break

    triggered = False
    for i in range(combined_df.shape[0]):
        high = combined_df.iloc[i]["High"]
        low = combined_df.iloc[i]["Low"]

        if order_type == "buy_limit" and low <= entry:
            triggered = True
            break
        elif order_type == "sell_limit" and high >= entry:
            triggered = True
            break
        elif order_type == "buy_stop" and high >= entry:
            triggered = True
            break
        elif order_type == "sell_stop" and low <= entry:
            triggered = True
            break
        elif order_type == "buy_stop_limit":
            if current_price < order["trigger_price"]:
                if high >= order["trigger_price"] and high >= entry:
                    triggered = True
                    break
            if current_price > order["trigger_price"]:
                if low <= order["trigger_price"] and high >= entry:
                    triggered = True
                    break
        elif order_type == "sell_stop_limit":
            if current_price < order["trigger_price"]:
                if high >= order["trigger_price"] and low <= entry:
                    triggered = True
                    break
            if current_price > order["trigger_price"]:
                if low <= order["trigger_price"] and low <= entry:
                    triggered = True
                    break

    if not triggered:
        return [f"cancel {order_id}"]

    # Volatility and range calculations
    price_range = combined_df["High"].max() - combined_df["Low"].min()
    avg_range = combined_df["High"].mean() - combined_df["Low"].mean()
    volatility = price_range / entry

    actions = []

    # Adjust TP down for low volatility
    if volatility < 0.002:
        new_tp_pips = int(tp_pips * 0.7)
        new_rr = new_tp_pips / sl_pips
        if new_rr >= min_rr:
            new_tp = calculate_target(entry, new_tp_pips, order_type, "tp")
            actions.append((f"move_tp {order_id} {round(new_tp, 3)}", (0.002 - volatility) * 1000))

    # Adjust TP up for high volatility
    if volatility > 0.01:
        new_tp_pips = int(tp_pips * 1.3)
        new_rr = new_tp_pips / sl_pips
        if new_rr >= min_rr:
            new_tp = calculate_target(entry, new_tp_pips, order_type, "tp")
            actions.append((f"move_tp {order_id} {round(new_tp, 3)}", (volatility - 0.01) * 1000))

    # Adjust SL farther if avg range is wide
    if avg_range > 0.5 * sl_pips * 0.01:
        new_sl_pips = int(sl_pips * 1.5)
        new_rr = tp_pips / new_sl_pips
        if new_rr >= min_rr:
            new_sl = calculate_target(entry, new_sl_pips, order_type, "sl")
            actions.append((f"move_sl {order_id} {round(new_sl, 3)}", (avg_range - 0.5 * sl_pips * 0.01) * 1000))

    # Adjust SL closer if RR is too high
    if rr > 4:
        new_sl_pips = int(sl_pips * 0.5)
        new_rr = tp_pips / new_sl_pips
        if new_rr >= min_rr:
            new_sl = calculate_target(entry, new_sl_pips, order_type, "sl")
            actions.append((f"move_sl {order_id} {round(new_sl, 3)}", (rr - 4) * 10))

    # Adjust Entry if price moved far from original
    last_price = combined_df.iloc[-1]["Close"]
    entry_gap = abs(last_price - entry)
    if entry_gap > 0.002 * entry:
        direction = -1 if order_type == "buy" else 1
        new_entry = round(entry + direction * (entry_gap / 2), 3)
        actions.append((f"move_entry {order_id} {new_entry}", entry_gap * 1000))

    if not actions:
        return [f"hold {order_id}"]

    # Sort actions by strength and return top 1–2
    actions.sort(key=lambda x: x[1], reverse=True)
    return [a[0] for a in actions[:2]]
        
if __name__ == "__main__":
    device = torch.device("cuda:0")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.float16, device_map={"": 0}
    ).to(device)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    processor_custom = Qwen2_5_VLProcessor_TS.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    folder = "/home/trangndp/projects/trading_bot/create_dataset_rs_rl/csv/XAUUSD"
    image_folder = "/home/trangndp/projects/trading_bot/create_dataset_rs_rl/image/XAUUSD"

    symbol = "XAUUSD"
    result_df_name = "xauusd_synthetic_orders_w_reasoning.csv"
    result_df = pd.DataFrame(columns=["prompt", "csv_file_path", "image_file_path", "answer"])

    for filename in os.listdir(folder):
        print(filename)
        pending_orders = {}
        ground_truth_actions = "<action>\n"
        HISTORY_ORDER_PROMPTS = "### Last 5 Orders:\n"
        PENDING_ORDER_PROMPTS = "### Pending Orders:\n"

        file_path_current = os.path.join(folder, filename)
        image_file_path_current = os.path.join(image_folder, f"{filename.split('.')[0]}.png")
        df_current = pd.read_csv(file_path_current, index_col=0)
        df_current = df_current[['Open', 'High', 'Low', 'Close', 'Volume']]

        sub_ts_id = int(filename.split('.')[0].split('_')[-1])
        period = filename.split('.')[0].split('_')[-2]
        prev_sub_ts_id = sub_ts_id - 1
        sub_filename = f"XAUUSDm_{period}_{prev_sub_ts_id}.csv"
        sub_file_path = os.path.join(folder, sub_filename)
        try:
            df_prev = pd.read_csv(sub_file_path, index_col=0)
            df_prev = df_prev[['Open', 'High', 'Low', 'Close', 'Volume']]
            df = pd.concat([df_current, df_prev])
        except:
            df = df_current

        min_low_pips = -50
        max_low_pips = -5
        min_high_pips = 5
        max_high_pips = 50

        min_percentage = 0.5
        max_percentage = 0.9
        split_percentage = split_percentage = random.uniform(min_percentage, max_percentage)

        num_ts = df.shape[0]
        split_point = int(num_ts * split_percentage)

        num_history_orders = 5
        picked_history_ids = random.sample(range(0, split_point), k=num_history_orders)
        picked_history_entrys = [df.iloc[i]["Open"] for i in picked_history_ids]

        max_num_pending = 10
        num_pending_orders = random.randint(0, max_num_pending)
        picked_pending_ids = random.sample(range(split_point, num_ts), k=num_pending_orders)
        current_pending_price = df.iloc[split_point - 1]["Open"]
        picked_pending_entrys = [df.iloc[i]["Open"] for i in picked_pending_ids]

        history_order_names = {0: "buy", 1: "sell"}
        history_order_id = [0, 1] 

        pending_order_names = {0: "buy_limit", 1: "sell_limit", 2: "buy_stop", 3: "sell_stop", 4: "buy_stop_limit", 5: "sell_stop_limit"}
        pending_order_id = [0, 1, 2, 3, 4, 5] 

        order_history_types = [random.choice(history_order_id) for _ in range(num_history_orders)]
        order_pending_types = [random.choice(pending_order_id) for _ in range(max_num_pending)]

        min_capital = 100
        max_capital = 100000
        picked_capital = random.sample(range(min_capital, max_capital + 1), k=1)

        min_risk_percent = 0.5
        max_risk_percent = 2
        picked_history_risk_percent = [random.uniform(min_risk_percent, max_risk_percent) for _ in range(num_history_orders)]
        picked_pending_risk_percent = [random.uniform(min_risk_percent, max_risk_percent) for _ in range(num_pending_orders)]

        min_ratio = 2
        max_ratio = 20
        picked_history_ratios = random.sample(range(min_ratio, max_ratio + 1), k=num_history_orders)
        picked_pending_ratios = random.sample(range(min_ratio, max_ratio + 1), k=num_pending_orders)

        percent_history_list = get_percent_at_value(min_ratio, max_ratio, picked_history_ratios)
        percent_pending_list = get_percent_at_value(min_ratio, max_ratio, picked_pending_ratios)

        min_tp_pips = 100
        max_tp_pips = 1000
        tp_history_pip_list = get_value_at_percent(min_tp_pips, max_tp_pips, percent_history_list)
        tp_pending_pip_list = get_value_at_percent(min_tp_pips, max_tp_pips, percent_pending_list)

        sl_history_pip_list = [tp_history_pip_list[i]/picked_history_ratios[i] for i in range(num_history_orders)]
        sl_pending_pip_list = [tp_pending_pip_list[i]/picked_pending_ratios[i] for i in range(num_pending_orders)]

        for i in range(num_history_orders):
            take_profit_price = calculate_target(float(picked_history_entrys[i]), tp_history_pip_list[i], history_order_names[order_history_types[i]], "tp")
            stop_loss_price = calculate_target(float(picked_history_entrys[i]), sl_history_pip_list[i], history_order_names[order_history_types[i]], "sl")
            is_win = check_order_result(folder, df, sub_ts_id, period, picked_history_ids[i], history_order_names[order_history_types[i]], take_profit_price, stop_loss_price)

            if is_win is not None:
                lot_size = calculate_lot_size(picked_capital[0], picked_history_risk_percent[i], sl_history_pip_list[i])

                if is_win:
                    profit = calculate_profit_or_loss(lot_size, tp_history_pip_list[i])
                else:
                    profit = - calculate_profit_or_loss(lot_size, sl_history_pip_list[i])

                picked_capital[0] = picked_capital[0] + (profit)
                
                HISTORY_ORDER_PROMPTS += f"[{i+1}] order_type={history_order_names[order_history_types[i]]}, entry={float(picked_history_entrys[i])}, take_profit_price={take_profit_price}, stop_loss_price={stop_loss_price}, take_profit_pips={tp_history_pip_list[i]}, stop_loss_pips={sl_history_pip_list[i]}, volume={lot_size}, result={'Win' if is_win else 'Lose'}, profit={profit}, balance={picked_capital[0]}\n"
            
        for i in range(num_pending_orders):
            pending_order = {"order_type": pending_order_names[order_pending_types[i]],
                            "entry": float(picked_pending_entrys[i]),
                            "trigger_price": float(picked_pending_entrys[i]) + random.uniform(min_low_pips, max_high_pips) if "stop_limit" else None,
                            "take_profit_price": calculate_target(float(picked_pending_entrys[i]), tp_pending_pip_list[i], pending_order_names[order_pending_types[i]], "tp"),
                            "stop_loss_price": calculate_target(float(picked_pending_entrys[i]), sl_pending_pip_list[i], pending_order_names[order_pending_types[i]], "sl"),
                            "take_profit_pips": tp_pending_pip_list[i],
                            "stop_loss_pips": sl_pending_pip_list[i],
                            "volume": calculate_lot_size(picked_capital[0], picked_pending_risk_percent[i], sl_pending_pip_list[i])
            }

            PENDING_ORDER_PROMPTS += f"[{i+1}] order_type={pending_order_names[order_pending_types[i]]}, entry={float(picked_pending_entrys[i])}, take_profit_price={take_profit_price}, stop_loss_price={stop_loss_price}, take_profit_pips={tp_pending_pip_list[i]}, stop_loss_pips={sl_pending_pip_list[i]}, volume={lot_size}\n"

            ground_truth_action = determine_action_for_pending_order(folder, pending_order, i+1, df, current_pending_price, picked_pending_ids[i], sub_ts_id, period, lookahead=2)
            for act in ground_truth_action:
                ground_truth_actions += act + "\n"

        ground_truth_actions = ground_truth_actions + "</action>"

        messages_reasoning = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_file_path_current
                    },
                    {
                        "type": "text",
                        "text": (
                            f"This is an {period} timeframe chart of {symbol}.\n"
                            "Please analyze it using:\n"
                            "- Elliott Wave: Label the wave count and identify which stage the market is in.\n"
                            "- ICT concepts: Look for liquidity sweeps, FVGs, OBs, and any major MSS.\n"
                            "- Classic tools: Use trendlines, Fibonacci zones, or divergence with RSI if relevant.\n\n"
                            "I want detailed reasoning for each observation, and a possible short-term forecast."
                        )
                    }
                ]
            }
        ]

        text_reasoning = processor.apply_chat_template(
            messages_reasoning, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages_reasoning)

        inputs = processor(
            text=[text_reasoning],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,        
                temperature=0.7,            
                top_p=0.95,             
                do_sample=True,           
                repetition_penalty=1.1,    
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        reasoning_part =  "\n".join(output_text) + "</think>"

        messages_input = [
            SYSTEM_PROMPT,
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"### SYMBOL: {symbol}\n"},
                    {"type": "text", "text": f"### PERIOD: {period}\n"},
                    {"type": "text", "text": f"### Account Balance: {picked_capital[0]}\n"},
                    {"type": "text", "text": "### Time Series Embedding:"},
                    {
                        "type": "multi_timeseries",
                        "multi_timeseries": "/home/trangndp/projects/trading_bot/dataset/BTCUSDm_infer.csv",
                    },
                    {"type": "text", "text": "\n"},
                    {"type": "text", "text": HISTORY_ORDER_PROMPTS},
                    {"type": "text", "text": PENDING_ORDER_PROMPTS},
                ],
            }
        ]

        text_input = processor_custom.apply_chat_template(
            messages_input, tokenize=False, add_generation_prompt=True
        )

        excecution_type = {0: "wait", 1: "instance", 2:'pending'}
        excecution_id = [0, 1, 2] 
        excecution_future_types = random.choice(excecution_id)
        if excecution_type[excecution_future_types] == "wait":
            orders = "<order>" + excecution_type[excecution_future_types] + "</order>"
        elif excecution_type[excecution_future_types] == "instance":
            not_is_win_count = 0
            while True:
                order_future_types = random.choice(history_order_id)

                min_risk_percent = 0.5
                max_risk_percent = 2
                picked_future_risk_percent = random.uniform(min_risk_percent, max_risk_percent) 

                min_ratio = 2
                max_ratio = 20
                picked_future_ratios = random.sample(range(min_ratio, max_ratio + 1), k=1)
                percent_future_list = get_percent_at_value(min_ratio, max_ratio, picked_future_ratios)

                min_tp_pips = 100
                max_tp_pips = 1000
                tp_future_pip_list = get_value_at_percent(min_tp_pips, max_tp_pips, percent_future_list)

                sl_future_pip_list = [tp_future_pip_list[0]/picked_future_ratios[0]]

                picked_future_entrys = float(df_current.iloc[-1]["Open"])

                take_profit_price = calculate_target(picked_future_entrys, tp_future_pip_list[0], history_order_names[order_future_types], "tp")
                stop_loss_price = calculate_target(picked_future_entrys, sl_future_pip_list[0], history_order_names[order_future_types], "sl")
                is_win = check_instance_future_order_result(folder, sub_ts_id, period, history_order_names[order_future_types], take_profit_price, stop_loss_price)

                if is_win is not None:
                    lot_size = calculate_lot_size(picked_capital[0], picked_future_risk_percent, sl_future_pip_list[0])

                    if is_win:
                        orders = f"<order>instance {history_order_names[order_future_types]} {take_profit_price} {stop_loss_price} {lot_size}</order>"
                        break

                not_is_win_count += 1
                if not_is_win_count == 10:
                    orders = f"<order>instance {history_order_names[order_future_types]} {take_profit_price} {stop_loss_price} {lot_size}</order>"
                    break
                
        elif excecution_type[excecution_future_types] == "pending":
            not_is_win_count = 0
            while True:
                order_future_types = random.choice(pending_order_id)

                min_risk_percent = 0.5
                max_risk_percent = 2
                picked_future_risk_percent = random.uniform(min_risk_percent, max_risk_percent) 

                min_ratio = 2
                max_ratio = 20
                picked_future_ratios = random.sample(range(min_ratio, max_ratio + 1), k=1)
                percent_future_list = get_percent_at_value(min_ratio, max_ratio, picked_future_ratios)

                min_tp_pips = 100
                max_tp_pips = 1000
                tp_future_pip_list = get_value_at_percent(min_tp_pips, max_tp_pips, percent_future_list)
                sl_future_pip_list = [tp_future_pip_list[0]/picked_future_ratios[0]]

                picked_future_entrys = float(df_current.iloc[-1]["Open"])

                if pending_order_names[order_future_types] == "buy_limit" or pending_order_names[order_future_types] == "sell_stop":
                    picked_future_diff = random.uniform(min_low_pips, max_low_pips) 
                    trigger_price = None
                    entry = picked_future_entrys + picked_future_diff
                elif pending_order_names[order_future_types] == "sell_limit" or pending_order_names[order_future_types] == "buy_stop":
                    picked_future_diff = random.uniform(min_high_pips, max_high_pips) 
                    trigger_price = None
                    entry = picked_future_entrys + picked_future_diff
                else:
                    low_range = random.uniform(min_low_pips, max_low_pips)
                    high_range = random.uniform(min_high_pips, max_high_pips)
                    trigger_future_diff = random.choice([low_range, high_range])
                    if pending_order_names[order_future_types] == "buy_stop_limit":
                        limit_future_diff = random.uniform(min_high_pips, max_high_pips) 
                        while trigger_future_diff > limit_future_diff:
                            limit_future_diff = random.uniform(min_high_pips, max_high_pips) 

                        trigger_price = picked_future_entrys + trigger_future_diff
                        entry = picked_future_entrys + limit_future_diff
                    if pending_order_names[order_future_types] == "sell_stop_limit":
                        limit_future_diff = random.uniform(min_low_pips, max_low_pips) 
                        while trigger_future_diff < limit_future_diff:
                            limit_future_diff = random.uniform(min_low_pips, max_low_pips)

                        trigger_price = picked_future_entrys + trigger_future_diff
                        entry = picked_future_entrys + limit_future_diff

                take_profit_price = calculate_target(picked_future_entrys, tp_future_pip_list[0], pending_order_names[order_future_types], "tp")
                stop_loss_price = calculate_target(picked_future_entrys, sl_future_pip_list[0], pending_order_names[order_future_types], "sl")
                is_win = check_pending_future_order_result(folder, sub_ts_id, period, picked_future_entrys, pending_order_names[order_future_types], trigger_price, entry, take_profit_price, stop_loss_price)

                if is_win is not None:
                    lot_size = calculate_lot_size(picked_capital[0], picked_future_risk_percent, sl_future_pip_list[0])

                    if is_win:
                        if "stop_limit" in pending_order_names[order_future_types]:
                            orders = f"<order>pending {pending_order_names[order_future_types]} {trigger_price} {entry} {take_profit_price} {stop_loss_price} {lot_size}</order>"
                        else:
                            orders = f"<order>pending {pending_order_names[order_future_types]} {entry} {take_profit_price} {stop_loss_price} {lot_size}</order>"
                        break

                not_is_win_count += 1
                if not_is_win_count == 10:
                    if "stop_limit" in pending_order_names[order_future_types]:
                        orders = f"<order>pending {pending_order_names[order_future_types]} {trigger_price} {entry} {take_profit_price} {stop_loss_price} {lot_size}</order>"
                    else:
                        orders = f"<order>pending {pending_order_names[order_future_types]} {entry} {take_profit_price} {stop_loss_price} {lot_size}</order>"
                    break

        result_df.loc[len(result_df)] = {"prompt": text_input + "\n<think>",
                                         "csv_file_path": file_path_current,
                                        "image_file_path": image_file_path_current,
                                        "answer": reasoning_part + "\n" + orders + "\n" + ground_truth_actions
        }

        result_df.to_csv(f"/home/trangndp/projects/trading_bot/create_dataset_rs_rl/{result_df_name}", index = True)
        del inputs, generated_ids, generated_ids_trimmed, output_text
        torch.cuda.empty_cache()






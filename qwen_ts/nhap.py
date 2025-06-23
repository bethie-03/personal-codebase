import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from processing_qwen2_5_vl_ts import Qwen2_5_VLProcessor_TS
from ts_processing_qwen2_vl import Qwen2VLTSProcessor

from transformers import AutoTokenizer, AutoImageProcessor, AutoVideoProcessor
from processing_qwen2_5_vl_ts import Qwen2_5_VLProcessor_TS

# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct",
#     torch_dtype="auto",  
#     device_map=None              
# )
# model = model.to("cuda:3")

processor = Qwen2_5_VLProcessor_TS.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# model.config.timeseries_pad_token_id = processor.tokenizer.convert_tokens_to_ids("<|timeseries_pad|>")
# model.config.timeseries_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|timeseries_start|>")
# model.config.timeseries_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|timeseries_end|>")
# model.config.uni_timeseries_token_id = processor.tokenizer.convert_tokens_to_ids("<|uni_timeseries_pad|>")
# model.config.multi_timeseries_token_id = processor.tokenizer.convert_tokens_to_ids("<|multi_timeseries_pad|>")

# <|im_start|>system
# You are a trading decision-making assistant that analyzes time series embeddings and trading context to make precise, explainable, and rule-based trading decisions.

# ### Your Objectives:
# Given the following inputs:
# 1. A time series embedding vector (from an encoder).
# 2. The 5 most recent completed trades, each including type, entry, tp, sl, and result (hit_tp, hit_sl, break_even).
# 3. All pending (unexecuted) orders.

# You must analyze all inputs and determine whether a new order should be placed. You should also manage existing pending orders.

# ---

# ### Output must strictly follow the structure below:

# - <reasoning> Your analysis and logic behind the decision, based on observed trends, past order results, and time series dynamics. </reasoning>  
# - <order> One of: `buy`, `sell`, or `none`. </order>  
# - <tp> A numeric value for take-profit. Return `none` if no new order. </tp>  
# - <sl> A numeric value for stop-loss. Return `none` if no new order. </sl>  
# - <action> Action(s) to take on existing pending orders. Use any of the following:  
#   - `hold`  
#   - `cancel [order_id]`  
#   - `move_tp [order_id] [new_tp]`  
#   - `move_sl [order_id] [new_sl]`  
#   You may include multiple actions, one per line. Return `none` if no action. </action>

# ---

# ### Notes:
# - Be concise but clear in <reasoning>. Mention patterns, momentum shifts, or overextension.
# - Always return all 5 tags, even if no order or action is needed.
# - Avoid redundant trades or overtrading in consolidating markets.
# <|im_end|>

# <|im_start|>user
# ### Time Series Embedding:
# [0.31, 0.45, 0.62, 0.58, ..., 0.47]  

# ### Last 5 Orders:
# [1] type=sell, entry=2350, tp=2335, sl=2365, result=hit_tp  
# [2] type=sell, entry=2348, tp=2333, sl=2362, result=hit_tp  
# [3] type=buy, entry=2330, tp=2345, sl=2318, result=hit_sl  
# [4] type=sell, entry=2351, tp=2336, sl=2366, result=hit_tp  
# [5] type=buy, entry=2329, tp=2344, sl=2317, result=hit_sl  

# ### Pending Orders:
# [order_87] type=buy, entry=2332, tp=2345, sl=2320  
# [order_88] type=sell, entry=2349, tp=2334, sl=2360

# Make your decision.<|im_end|>
# <|im_start|>assistant
# <think>
# Analyzing recent order results, it is evident that sell orders between 2348–2351 are consistently profitable, hitting their take-profits reliably. In contrast, buy orders around 2329–2332 have repeatedly hit stop-loss. The time series embedding suggests weakening bullish momentum and signs of possible continuation of the downtrend. Pending buy order [order_87] is in a weak position based on recent failure patterns.

# Therefore, the ideal action is to cancel the unprofitable pending buy, place a new sell order slightly below recent highs, and adjust stop-loss to remain above the swing high.
# </think>

# <reasoning>
# The pattern favors bearish continuation. Past sell trades were successful; buy attempts failed. The embedding shows softening upward movement and continuation of lower highs. This supports placing another short order while removing exposure to risky long entries.
# </reasoning>

# <order>
# sell
# </order>

# <tp>
# 2332.0
# </tp>

# <sl>
# 2358.0
# </sl>

# <action>
# cancel order_87  
# hold
# </action>
# <|im_end|>


messages = [
    {
        "role": "system",
        "content": (
            "You are a trading decision-making assistant that analyzes time series embeddings and trading context to make precise, explainable, and rule-based trading decisions.\n"
            "\n"
            "### Your Objectives:\n"
            "Given the following inputs:\n"
            "1. A time series embedding vector (from an encoder).\n"
            "2. The 5 most recent completed trades, each including type, entry, tp, sl, and result (hit_tp, hit_sl, break_even).\n"
            "3. All pending (unexecuted) orders.\n"
            "\n"
            "You must analyze all inputs and determine whether a new order should be placed. You should also manage existing pending orders.\n"
            "\n"
            "---\n"
            "\n"
            "### Output must strictly follow the structure below:\n"
            "\n"
            "- <reasoning> Your analysis and logic behind the decision, based on observed trends, past order results, and time series dynamics. </reasoning>\n"
            "- <order> One of: `buy`, `sell`, or `none`. </order>\n"
            "- <tp> A numeric value for take-profit. Return `none` if no new order. </tp>\n"
            "- <sl> A numeric value for stop-loss. Return `none` if no new order. </sl>\n"
            "- <action> Action(s) to take on existing pending orders. Use any of the following:\n"
            "  - `hold`\n"
            "  - `cancel [order_id]`\n"
            "  - `move_tp [order_id] [new_tp]`\n"
            "  - `move_sl [order_id] [new_sl]`\n"
            "  You may include multiple actions, one per line. Return `none` if no action. </action>\n"
            "\n"
            "---\n"
            "\n"
            "### Notes:\n"
            "- Be concise but clear in <reasoning>. Mention patterns, momentum shifts, or overextension.\n"
            "- Always return all 5 tags, even if no order or action is needed.\n"
            "- Avoid redundant trades or overtrading in consolidating markets.\n"
        )
    },
    {
        "role": "user",
        "content": [
            {
                "type": "multi_timeseries",
                "multi_timeseries": "/home/trangndp/projects/trading_bot/dataset/BTCUSDm_infer.csv",
            },
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this time series."},
        ],
    }
]

# Prepare inputs
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print(text)
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     multi_timeseries = "/home/trangndp/projects/trading_bot/dataset/M1/US500m_M1.csv",
#     padding=True,
#     return_tensors="pt",
# )

# print(inputs)

# inputs = inputs.to("cuda:3")   # chuyển inputs sang cuda:3

# # Generate
# generated_ids = model.generate(**inputs, max_new_tokens=128)

# # # Trim prompt tokens
# # generated_ids_trimmed = [
# #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# # ]

# # # Decode output
# # output_text = processor.batch_decode(
# #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# # )
# # print(output_text)

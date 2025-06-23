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

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype="auto",  
    device_map=None              
)
model = model.to("cuda:3")

processor = Qwen2_5_VLProcessor_TS.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

model.config.timeseries_pad_token_id = processor.tokenizer.convert_tokens_to_ids("<|timeseries_pad|>")
model.config.timeseries_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|timeseries_start|>")
model.config.timeseries_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|timeseries_end|>")
model.config.uni_timeseries_token_id = processor.tokenizer.convert_tokens_to_ids("<|uni_timeseries_pad|>")
model.config.multi_timeseries_token_id = processor.tokenizer.convert_tokens_to_ids("<|multi_timeseries_pad|>")
model.config.multi_timeseries_token_id = processor.tokenizer.convert_tokens_to_ids("<think>")
model.config.multi_timeseries_token_id = processor.tokenizer.convert_tokens_to_ids("</think>")

messages = [
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

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    multi_timeseries = "/home/trangndp/projects/trading_bot/dataset/M1/US500m_M1.csv",
    padding=True,
    return_tensors="pt",
)

print(inputs)

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

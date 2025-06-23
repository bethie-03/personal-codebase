import numpy as np
from typing import List, Optional, Union
from transformers.image_utils import ImageInput
from transformers.video_utils import VideoInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.processing_utils import ImagesKwargs, MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from .ts_processing_qwen2_vl import Qwen2VLTSProcessor
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor, Qwen2_5_VLProcessorKwargs

class Qwen2_5_VLProcessor_TS(Qwen2_5_VLProcessor):
    
    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        special_tokens = {
            "additional_special_tokens": [
                "<|timeseries_pad|>",
                "<|timeseries_start|>",
                "<|timeseries_end|>",
                "<|uni_timeseries_pad|>",
                "<|multi_timeseries_pad|>",
                "<think>",
                "</think>"
            ]   
        }
        tokenizer.add_special_tokens(special_tokens)

        chat_template = """
        {% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% set uni_timeseries_count = namespace(value=0) %}{% set multi_timeseries_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system
        You are a helpful assistant.<|im_end|>
        {% endif %}<|im_start|>{{ message['role'] }}
        {% if message['content'] is string %}{{ message['content'] }}<|im_end|>
        {% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif content['type'] == 'uni_timeseries' %}{% set uni_timeseries_count.value = uni_timeseries_count.value + 1 %}{% if add_vision_id %}Univariate Timeseries {{ uni_timeseries_count.value }}: {% endif %}<|timeseries_start|><|uni_timeseries_pad|><|timeseries_end|>{% elif content['type'] == 'multi_timeseries' %}{% set multi_timeseries_count.value = multi_timeseries_count.value + 1 %}{% if add_vision_id %}Multivariate Timeseries {{ multi_timeseries_count.value }}: {% endif %}<|timeseries_start|><|multi_timeseries_pad|><|timeseries_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>
        {% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
        {% endif %}
        """

        self.uni_timeseries_token = "<|uni_timeseries_pad|>" if not hasattr(tokenizer, "uni_timeseries_token") else tokenizer.uni_timeseries_token
        self.multi_timeseries_token = "<|multi_timeseries_pad|>" if not hasattr(tokenizer, "multi_timeseries_token") else tokenizer.multi_timeseries_token
        self.uni_timeseries_token_id = (
            tokenizer.uni_timeseries_token_id
            if getattr(tokenizer, "uni_timeseries_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.uni_timeseries_token)
        )
        self.multi_timeseries_token_id = (
            tokenizer.multi_timeseries_token_id
            if getattr(tokenizer, "multi_timeseries_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.multi_timeseries_token)
        )
        self.timeseries_processor = Qwen2VLTSProcessor(features="MS")

        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        uni_timeseries: str = None,
        multi_timeseries: str = None,
        **kwargs: Unpack[Qwen2_5_VLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
            - **second_per_grid_ts** -- List of video seconds per time grid. Returned when `videos` is not `None`.
        """

        if uni_timeseries is not None and multi_timeseries is not None:
            raise TypeError("Only one of 'uni_timeseries' or 'multi_timeseries' should be provided, not both.")

        output_kwargs = self._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = videos_inputs = uni_timeseries_inputs = multi_timeseries_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]

        if videos is not None:
            # pop fps in advance for passing kwargs validation
            fps = output_kwargs["videos_kwargs"].pop("fps", 2.0)

            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]

            if isinstance(fps, (int, float)):
                second_per_grid_ts = [self.video_processor.temporal_patch_size / fps] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                second_per_grid_ts = [self.video_processor.temporal_patch_size / tmp for tmp in fps]
            else:
                raise ValueError(
                    f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                )
            videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})

        if uni_timeseries is not None:
            uni_timeseries_inputs = self.timeseries_processor(data_path=uni_timeseries, return_tensors=kwargs.get("return_tensors", None))
            uni_timeseries_shape = uni_timeseries_inputs["timeseries_values"].shape
            if uni_timeseries_shape[1] != 1:
                raise ValueError("uni_timeseries must have more than one data column (excluding the datetime index).")

        if multi_timeseries is not None:
            multi_timeseries_inputs = self.timeseries_processor(data_path=multi_timeseries, return_tensors=kwargs.get("return_tensors", None))
            multi_timeseries_shape = multi_timeseries_inputs["timeseries_values"].shape
            if multi_timeseries_shape[1] < 2:
                raise ValueError("multi_timeseries must have more than one data column (excluding the datetime index).")

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place
        if images is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if videos is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    num_video_tokens = video_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.video_token, "<|placeholder|>" * num_video_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        if uni_timeseries is not None:
            merge_length = self.timeseries_processor.patch_len
            index = 0
            for i in range(len(text)):
                while self.uni_timeseries_token in text[i]:
                    num_timeseries_tokens = uni_timeseries_shape[0] // merge_length
                    text[i] = text[i].replace(self.uni_timeseries_token, "<|placeholder|>" * num_timeseries_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.uni_timeseries_token)

        if multi_timeseries is not None:
            merge_length = self.timeseries_processor.patch_len
            index = 0
            for i in range(len(text)):
                while self.multi_timeseries_token in text[i]:
                    num_timeseries_tokens = multi_timeseries_shape[0] // merge_length
                    text[i] = text[i].replace(self.multi_timeseries_token, "<|placeholder|>" * num_timeseries_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.multi_timeseries_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs, **uni_timeseries_inputs, **multi_timeseries_inputs}, tensor_type=return_tensors)


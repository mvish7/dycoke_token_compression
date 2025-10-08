import argparse
import time
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor, TextStreamer
from transformers import Qwen2_5_VLForConditionalGeneration
# custom imports
from vlm_runners.vlms_base import BaseVLInference
from customized_qwen2_5_vl.configuration_qwen2_5_vl import CustomQwen2_5_VLConfig
from customized_qwen2_5_vl.modeling_qwen2_5_vl import CustomQwen2_5_VLForConditionalGeneration
from dycoke.prunable_dynamic_cache import PrunableDynamicCache
from utils.construct_input_message import construct_qwen2_5_vl_message
from qwen_vl_utils import process_vision_info


class Qwen2_5_Vl_Inference(BaseVLInference):
    """
    Inference wrapper for Qwen2.5-VL model.
    """
    def __init__(self, model_id: str, dycoke_args: dict):
        # Specific token frame count for Qwen-VL
        dycoke_args["dycoke_num_tokens_per_frame"] = 360
        super().__init__(
            model_id, dycoke_args,
            CustomQwen2_5_VLConfig, CustomQwen2_5_VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
        )

    def prepare_inputs(self, prompt: str, video_path: str, fps: float, **kwargs):
        """
        Qwen2.5-VL specific input message construction and tokenization.
        """

        message = construct_qwen2_5_vl_message(prompt, video_path, fps)
        text = self.processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)

        image_inputs, video_inputs = process_vision_info(message)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=fps,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device) # Use model.device for multi-GPU support

        return inputs, inputs["input_ids"].shape[-1]
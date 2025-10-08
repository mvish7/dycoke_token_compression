import argparse
import time
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor, TextStreamer
from transformers import Gemma3ForConditionalGeneration, Gemma3Config
# Custom imports
from vlm_runners.vlms_base import BaseVLInference
from customized_gemma3.configuration_gemma3 import CustomGemma3Config
from customized_gemma3.modeling_gemma3 import CustomGemma3ForConditionalGeneration
from utils.construct_input_message import construct_gemma3_message
from utils.video_reader import read_video_as_images
from dycoke.prunable_dynamic_cache import PrunableDynamicCache


class Gemma3Inference(BaseVLInference):
    """Inference wrapper for Gemma3 model."""
    def __init__(self, model_id: str, dycoke_args: dict):
        # Specific token frame count for Gemma3
        dycoke_args["dycoke_num_tokens_per_frame"] = 256
        super().__init__(
            model_id, dycoke_args,
            CustomGemma3Config, CustomGemma3ForConditionalGeneration, Gemma3ForConditionalGeneration
        )

    def prepare_inputs(self, prompt: str, video_path: str, fps: float = 1.0, sampling_factor: int = 4):
        """Gemma3 specific input message construction and tokenization."""
        # Custom Imports (as seen in original Gemma3 code)
        # from customized_gemma3.configuration_gemma3 import CustomGemma3Config
        # from customized_gemma3.modeling_gemma3 import CustomGemma3ForConditionalGeneration
        # from utils.construct_input_message import construct_gemma3_message
        # from utils.video_reader import read_video_as_images

        images = read_video_as_images(video_path, fps=fps, sampling_factor=sampling_factor)
        message = construct_gemma3_message(prompt, images)

        inputs = self.processor.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        return inputs, inputs["input_ids"].shape[-1]
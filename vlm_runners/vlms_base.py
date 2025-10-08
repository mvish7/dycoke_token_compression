import argparse
import time
from abc import ABC, abstractmethod
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor, TextStreamer, PreTrainedModel, PreTrainedTokenizerBase
from dycoke.prunable_dynamic_cache import PrunableDynamicCache


class BaseVLInference(ABC):
    """Abstract base class for running Video-Language model inference with DyCoke."""

    def __init__(self, model_id: str, dycoke_args: dict,
                 custom_config_class, custom_model_class, original_model_class):
        """
        Initialize the model, processor, and configurations.

        Args:
            model_id (str): Hugging Face model ID.
            dycoke_args (dict): DyCoke pruning arguments.
            custom_config_class: The customized configuration class.
            custom_model_class: The customized model class.
            original_model_class: The original model class from transformers.
        """
        # register with auto classes from transformers
        AutoConfig.register(f"custom_{custom_model_class.__name__.lower()}", custom_config_class)
        AutoModel.register(custom_config_class, custom_model_class)

        # configuration and DyCoke Settings
        self.config = custom_config_class.from_pretrained(model_id)
        self._apply_dycoke_settings(dycoke_args)

        # Model Loading
        if dycoke_args["use_dycoke"]:
            self.model: PreTrainedModel = custom_model_class.from_pretrained(
                model_id,
                config=self.config,
                attn_implementation={"text_config": "eager", "vision_config": "sdpa"},
                device_map="auto",
                torch_dtype=torch.bfloat16,
            ).eval()
        else:
            self.model: PreTrainedModel = original_model_class.from_pretrained(
                model_id, device_map="auto", torch_dtype=torch.bfloat16
            ).eval()

        # Processor and Streamer
        self.processor: PreTrainedTokenizerBase = AutoProcessor.from_pretrained(model_id)
        self.streamer = TextStreamer(self.processor, skip_prompt=True, skip_special_tokens=True)

    def _apply_dycoke_settings(self, dycoke_args):
        """
        Configure DyCoke pruning parameters if enabled.
        """
        if dycoke_args["use_dycoke"]:
            self.config.dycoke = True
            self.config.dycoke_l = dycoke_args["dycoke_l"]
            self.config.dycoke_p = dycoke_args["dycoke_p"]
            # NOTE: dycoke_num_tokens_per_frame is model-specific, but here I set a default
            # and allow the derived class to adjust it in their init if needed.
            self.config.dycoke_num_tokens_per_frame = dycoke_args.get("dycoke_num_tokens_per_frame", 300)
            self.config.dycoke_k = dycoke_args["dycoke_k"]
        else:
            self.config.dycoke = False

    @abstractmethod
    def prepare_inputs(self, prompt: str, video_path: str, fps: float, **kwargs):
        """
        Abstract method to construct message and tokenize inputs.
        """
        pass

    def generate(self, inputs: dict, input_len: int, max_new_tokens: int = 512):
        """
        Generate response from the model, managing DyCoke cache.
        (Common logic)
        """
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated() / 1024 ** 3

        start_time = time.time()

        with torch.inference_mode():
            if self.config.dycoke:
                dycoke_cache = PrunableDynamicCache()
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    past_key_values=dycoke_cache,
                    output_attentions=True,
                    cache_implementation=None,
                    streamer=self.streamer,
                )
            else:
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    output_attentions=False,
                    streamer=self.streamer,
                )

        torch.cuda.synchronize()
        end_time = time.time()

        memory_after = torch.cuda.memory_allocated() / 1024 ** 3
        generation = generation[0][input_len:]

        return generation, end_time - start_time, memory_after - memory_before

    @staticmethod
    def log_metrics(generation, generation_time, memory_used):
        """Prints generation performance metrics."""
        tokens_per_second = len(generation) / generation_time
        print(f"\n--- Inference Summary ---")
        print(f"Generated {len(generation)} tokens in {generation_time:.2f}s")
        print(f"Speed: {tokens_per_second:.2f} tokens/second")
        print(f"Memory used: {memory_used:.2f} GB")
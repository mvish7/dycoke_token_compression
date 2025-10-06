import argparse
import time
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor, TextStreamer
from transformers import Qwen2_5_VLForConditionalGeneration
# custom imports
from customized_qwen2_5_vl.configuration_qwen2_5_vl import CustomQwen2_5_VLConfig
from customized_qwen2_5_vl.modeling_qwen2_5_vl import CustomQwen2_5_VLForConditionalGeneration
from dycoke.prunable_dynamic_cache import PrunableDynamicCache
from utils.construct_input_message import construct_qwen2_5_vl_message
from qwen_vl_utils import process_vision_info


class Qwen2_5_Vl_Inference:
    def __init__(self, model_id: str, dycoke_args: dict):
        """
        Initialize the model, processor, and configurations.

        Args:
            model_id (str): Hugging Face model ID.
            dycoke_args (dict): DyCoke pruning arguments.
        """
        AutoConfig.register("custom_qwen2_5_vl", CustomQwen2_5_VLConfig)
        AutoModel.register(CustomQwen2_5_VLConfig, CustomQwen2_5_VLForConditionalGeneration)

        self.config = CustomQwen2_5_VLConfig.from_pretrained(model_id)
        self._apply_dycoke_settings(dycoke_args)

        if dycoke_args["use_dycoke"]:
            self.model = CustomQwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                config=self.config,
                attn_implementation={"text_config": "eager", "vision_config": "sdpa"},
                device_map="auto",
                torch_dtype=torch.bfloat16,
            ).eval()
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, device_map="auto", torch_dtype=torch.bfloat16
            ).eval()

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.streamer = TextStreamer(self.processor, skip_prompt=True, skip_special_tokens=True)

    def _apply_dycoke_settings(self, dycoke_args):
        """Configure DyCoke pruning parameters if enabled."""
        if dycoke_args["use_dycoke"]:
            self.config.dycoke = True
            self.config.dycoke_l = dycoke_args["dycoke_l"]
            self.config.dycoke_p = dycoke_args["dycoke_p"]
            self.config.dycoke_num_tokens_per_frame = 360
            self.config.dycoke_k = dycoke_args["dycoke_k"]
        else:
            self.config.dycoke = False

    def prepare_inputs(self, prompt: str, video_path: str, fps: float= 1.0):
        """
        Construct message and tokenize inputs.

        Args:
            prompt (str): Text prompt.
            video_path (str): Path to video file.
            fps (float): Frames per second for video sampling.

        Returns:
            dict: Tokenized inputs for the model.
            int: Input length.
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
        inputs = inputs.to("cuda")

        return inputs, inputs["input_ids"].shape[-1]

    def generate(self, inputs: dict, input_len: int, max_new_tokens: int = 512):
        """
        Generate response from the model.

        Args:
            inputs (dict): Preprocessed inputs.
            input_len (int): Original input length.
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            torch.Tensor: Generated token IDs.
            float: Generation time in seconds.
            float: GPU memory used in GB.
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

def log_metrics(generation, generation_time, memory_used):
    """Prints generation performance metrics."""
    tokens_per_second = len(generation) / generation_time
    print(f"\n--- Inference Summary ---")
    print(f"Generated {len(generation)} tokens in {generation_time:.2f}s")
    print(f"Speed: {tokens_per_second:.2f} tokens/second")
    print(f"Memory used: {memory_used:.2f} GB")




def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with Custom Gemma3 model.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the model.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model ID.")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to sample video.")
    parser.add_argument("--sampling_factor", type=int, default=12, help="selects every Nth frame from the video.")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum new tokens to generate.")
    parser.add_argument("--use_dycoke", action="store_true", help="Disable DyCoke pruning.")
    parser.add_argument("--dycoke_p", type=float, default=0.7, help="Pruning ratio")
    parser.add_argument("--dycoke_l", type=int, default=3, help="Layer index to start DyCoke pruning")
    parser.add_argument("--dycoke_k", type=float, default=0.3, help="Merging ratio for temporal token merging")
    return parser.parse_args()


def main():
    args = parse_args()

    inference = Qwen2_5_Vl_Inference(args.model_id, {"use_dycoke": args.use_dycoke, "dycoke_l": args.dycoke_l,
                                                "dycoke_p": args.dycoke_p, "dycoke_k": args.dycoke_k})

    inputs, input_len = inference.prepare_inputs(args.prompt, args.video, fps=args.fps)

    generation, gen_time, mem_used = inference.generate(inputs, input_len, max_new_tokens=args.max_tokens)

    log_metrics(generation, gen_time, mem_used)


if __name__ == "__main__":
    main()


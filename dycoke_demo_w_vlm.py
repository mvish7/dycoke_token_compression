import argparse
from vlm_runners.vlms_base import BaseVLInference
from vlm_runners.qwen2_5_vl_runner import Qwen2_5_Vl_Inference
from vlm_runners.gemma3_runner import Gemma3Inference

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with a Custom VL model.")
    # Common arguments
    parser.add_argument("--video", type=str, required=True, help="Path to input video.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the model.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model ID.")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to sample video.")
    parser.add_argument("--sampling_factor", type=int, default=12,
                        help="Selects every Nth frame from the video (Gemma3 specific).")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum new tokens to generate.")
    # DyCoke arguments
    parser.add_argument("--use_dycoke", action="store_true", help="Enable DyCoke pruning.")
    parser.add_argument("--dycoke_p", type=float, default=0.7, help="Pruning ratio")
    parser.add_argument("--dycoke_l", type=int, default=3, help="Layer index to start DyCoke pruning")
    parser.add_argument("--dycoke_k", type=float, default=0.3, help="Merging ratio for temporal token merging")
    return parser.parse_args()


def main():
    args = parse_args()

    # Simple logic to select the correct class based on model_id prefix
    if "qwen" in args.model_id.lower():
        InferenceClass = Qwen2_5_Vl_Inference
    elif "gemma" in args.model_id.lower():
        InferenceClass = Gemma3Inference
    else:
        raise ValueError(f"Unsupported model ID: {args.model_id}")

    dycoke_args = {
        "use_dycoke": args.use_dycoke,
        "dycoke_l": args.dycoke_l,
        "dycoke_p": args.dycoke_p,
        "dycoke_k": args.dycoke_k
    }

    inference: BaseVLInference = InferenceClass(args.model_id, dycoke_args)

    # Pass all relevant arguments to prepare_inputs
    inputs, input_len = inference.prepare_inputs(
        args.prompt, args.video, fps=args.fps, sampling_factor=args.sampling_factor
    )

    generation, gen_time, mem_used = inference.generate(inputs, input_len, max_new_tokens=args.max_tokens)

    inference.log_metrics(generation, gen_time, mem_used)


if __name__ == "__main__":
    main()
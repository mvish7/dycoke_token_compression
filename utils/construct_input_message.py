import numpy as np


def construct_gemma3_message(text_prompt: str, images: np.ndarray):

    usr_content = []
    for img_id in range(len(images)):
        image = images[img_id]
        usr_content.append({"type": "image", "image": image})
    usr_content.append({"type": "text", "text": text_prompt})

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": usr_content
        }
    ]
    return messages


def construct_qwen2_5_vl_message(text_prompt: str, video_path: str, fps: float = 1.0):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps,
                },
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    return messages


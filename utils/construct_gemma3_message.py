import numpy as np


def construct_message(text_prompt: str, images: np.ndarray):

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

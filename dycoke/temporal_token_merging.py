"""
taken from https://github.com/KD-TAO/DyCoke
"""

import torch


def dycoke_ttm(image_feature, num_tokens_per_frame=256, merging_ratio=0.5):
    # Split frames into tokens
    num_frames = image_feature.shape[0]
    merging_ratio = 1 - merging_ratio
    image_feature = image_feature.view(image_feature.shape[0] * image_feature.shape[1], -1)
    # Calculate similarities between adjacent even frames
    similarities = []
    for i in range(0, num_frames - 1, 2):
        # Get tokens for adjacent frames
        frame1_tokens = image_feature[i * num_tokens_per_frame:(i + 1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i + 1) * num_tokens_per_frame:(i + 2) * num_tokens_per_frame]

        # Calculate cosine similarity between normalized tokens
        frame1_norm = torch.nn.functional.normalize(frame1_tokens, p=2, dim=1)
        frame2_norm = torch.nn.functional.normalize(frame2_tokens, p=2, dim=1)
        similarity = torch.nn.functional.cosine_similarity(frame1_norm, frame2_norm, dim=1)
        similarities.append(similarity)

    similarities = torch.stack([torch.tensor(similarity) for similarity in similarities])

    # Process even frames
    modified_image_feature = []
    for i in range(0, num_frames - 1, 2):
        frame1_tokens = image_feature[i * num_tokens_per_frame:(i + 1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i + 1) * num_tokens_per_frame:(i + 2) * num_tokens_per_frame]

        avg_similarity = similarities[i // 2]
        num_tokens_to_keep = int(merging_ratio * num_tokens_per_frame)
        tokens_to_keep = avg_similarity.topk(num_tokens_to_keep, largest=False).indices

        modified_image_feature.append(frame1_tokens)
        modified_image_feature.append(frame2_tokens[tokens_to_keep])

    # Process odd frames
    odd_similarities = []
    for i in range(0, num_frames - 4, 4):
        frame1_tokens = image_feature[i * num_tokens_per_frame:(i + 1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i + 2) * num_tokens_per_frame:(i + 3) * num_tokens_per_frame]

        similarity = torch.nn.functional.cosine_similarity(frame1_tokens, frame2_tokens, dim=1)
        odd_similarities.append(similarity)

    odd_similarities = torch.stack([torch.tensor(similarity) for similarity in odd_similarities])

    for i in range(0, num_frames - 4, 4):
        frame1_tokens = image_feature[i * num_tokens_per_frame:(i + 1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i + 2) * num_tokens_per_frame:(i + 3) * num_tokens_per_frame]

        avg_similarity = odd_similarities[i // 4]
        num_tokens_to_keep = int(merging_ratio * num_tokens_per_frame)
        tokens_to_keep = avg_similarity.topk(num_tokens_to_keep, largest=False).indices

        modified_image_feature[i] = frame1_tokens
        modified_image_feature[i + 2] = frame2_tokens[tokens_to_keep]

    # Combine all tokens
    combined_tokens = torch.cat(modified_image_feature, dim=0)
    return combined_tokens

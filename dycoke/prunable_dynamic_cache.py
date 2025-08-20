"""
taken from https://github.com/KD-TAO/DyCoke
"""

from typing import List, Tuple
import torch
from transformers.cache_utils import DynamicCache

class PrunableDynamicCache(DynamicCache):

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.kv_cache = None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        if self.kv_cache is None:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            return torch.gather(
                self.key_cache[layer_idx],
                dim=2,
                index=torch.tensor(self.kv_cache, device=self.key_cache[layer_idx].device)
                .view(1, 1, -1, 1)
                .expand(
                    self.key_cache[layer_idx].size(0),
                    self.key_cache[layer_idx].size(1),
                    -1,
                    self.key_cache[layer_idx].size(3),
                ),
            ), torch.gather(
                self.value_cache[layer_idx],
                dim=2,
                index=torch.tensor(self.kv_cache, device=self.value_cache[layer_idx].device)
                .view(1, 1, -1, 1)
                .expand(
                    self.value_cache[layer_idx].size(0),
                    self.value_cache[layer_idx].size(1),
                    -1,
                    self.value_cache[layer_idx].size(3),
                ),
            )

    def update_cache(self, image_attention, config):
        # Pre-calculate values to avoid repeated computation
        start_idx = config.image_token_start_index
        img_len = config.image_token_length
        num_keep = int(img_len * (1 - config.dycoke_radio))

        # Get top indices in one operation
        top_indices = torch.topk(image_attention, num_keep, sorted=False)[1] + start_idx

        # Create ranges efficiently using single arange call
        device = image_attention.device
        full_range = torch.arange(config.seq_length_with_past, device=device)
        keep_indexs = torch.cat([
            full_range[:start_idx],
            top_indices,
            full_range[start_idx + img_len:],
        ])

        # Convert to list once at end
        self.kv_cache = keep_indexs.tolist()

    def dycoke_pruning(self, attn, layer_idx, config):
        attention_avg = attn[1].mean(1)[0, -1]
        start_idx = config.image_token_start_index
        img_len = config.image_token_length
        image_attention = attention_avg[start_idx : start_idx + img_len]

        if config.attention_score is not None:
            config.similarity = torch.nn.functional.cosine_similarity(image_attention, config.attention_score, dim=0)
        else:
            config.similarity = 0
        config.attention_score = image_attention

        if config.similarity < 0.9:
            self.update_cache(image_attention, config)

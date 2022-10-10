import logging

import torch.nn as nn
from diffusers.models.attention import CrossAttention
from tfsdg.models.attention import StructuredCrossAttention

logger = logging.getLogger(__name__)


def replace_cross_attention(nn_module: nn.Module, name: str) -> None:
    for attr_str in dir(nn_module):
        target_attr = getattr(nn_module, attr_str)

        if isinstance(target_attr, CrossAttention):
            query_dim = target_attr.to_q.in_features
            assert target_attr.to_k.in_features == target_attr.to_v.in_features
            context_dim = target_attr.to_k.in_features
            heads = target_attr.heads
            dim_head = int(target_attr.scale**-2)
            dropout = target_attr.to_out[-1].p

            sca_kwargs = {
                "query_dim": query_dim,
                "context_dim": context_dim,
                "heads": heads,
                "dim_head": dim_head,
                "dropout": dropout,
            }

            if attr_str == "attn2":
                sca = StructuredCrossAttention(**sca_kwargs, struct_attention=True)
            else:
                sca = StructuredCrossAttention(**sca_kwargs, struct_attention=False)

            original_params = list(target_attr.parameters())
            proposed_params = list(sca.parameters())
            assert len(original_params) == len(proposed_params)

            for p1, p2 in zip(original_params, proposed_params):
                p2.data.copy_(p1.data)

            logger.info(f"Replaced: {name} {attr_str} to {sca}")
            setattr(nn_module, attr_str, sca)

    for name, immediate_child_module in nn_module.named_children():
        replace_cross_attention(nn_module=immediate_child_module, name=name)

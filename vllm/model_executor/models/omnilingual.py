
import torch
import torch.nn as nn
from typing import Annotated, Any, Literal, TypeAlias

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import SupportsMultiModal, SupportsPP


# Audio inputs here
class OmniFeatures(TensorSchema):
    """
    Dimensions:
        - na: Number of audios
        - nmb: Number of mel bins
    """

    type: Literal["audio_features"]
    input_features: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("na", "nmb", 3000),
    ]

    feature_attention_mask: Annotated[
        torch.Tensor,
        TensorShape("na", 3000),
    ]


class OmniEmbeddings(TensorSchema):
    """
    Dimensions:
        - bn: Batch size
        - naf: Number of audio features
        - hs: Hidden size (must match the hidden size of language model
          backbone)
    """

    type: Literal["audio_embeds"] = "audio_embeds"

    audio_embeds: Annotated[
        list[torch.Tensor],
        TensorShape("bn", "naf", "hs", dynamic_dims={"naf"}),
    ]

OmniInputs: TypeAlias = OmniFeatures | OmniEmbeddings


"""Inference omnilingual model"""
class OmnilingualForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            # TODO: verify special audio placeholder tokens
            pass

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        # TODO: Verify this...

    def validate_audio_input(self, **kwargs: object) -> OmniInputs | None:
        audio_embeds = kwargs.pop("audio_embeds", None)
        input_features = kwargs.pop("input_features", None)

        if audio_embeds is not None:
            return OmniEmbeddings(type="audio_embeds", audio_embeds=audio_embeds)

        feature_attention_mask = kwargs.pop("feature_attention_mask", None)

        if input_features is not None:
            return OmniFeatures(
                type="audio_features",
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
            )

        elif audio_embeds is None and input_features is None:
            return None

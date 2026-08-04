
import torch
import torch.nn as nn
from typing import Annotated, Any, Literal, TypeAlias

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import SupportsMultiModal, SupportsPP


# Audio inputs here
class OmnilingualAudioInputs(TensorSchema):
    """
    Dimensions:
        - na: Number of audio sequences
        - ns: Number of waveform samples
    """

    type: Literal["audio_values"] = "audio_values"

    input_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("na", "samples"),
    ]

    attention_mask: Annotated[
        torch.Tensor,
        TensorShape("na", "samples"),
    ]
class OmnilingualEmbeddingInputs(TensorSchema):
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

OmnilingualInputs : TypeAlias = OmnilingualAudioInputs | OmnilingualEmbeddingInputs 


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

    def _parse_and_validate_audio_input(self, **kwargs: object) -> OmnilingualInputs | None:
        audio_embeds = kwargs.pop("audio_embeds", None)
        input_values = kwargs.pop("input_values", None)

        if audio_embeds is not None:
            return OmnilingualEmbeddingInputs(type="audio_embeds", audio_embeds=audio_embeds)

        if input_values is not None:
            attention_mask = kwargs.pop("attention_mask", None)

            return OmnilingualAudioInputs(
                type="audio_values",
                input_values=input_values,
                attention_mask=attention_mask,
            )

        return None

    # def _process_audio_input(self, audio_input: Qwen2AudioInputs) -> torch.Tensor | tuple[torch.Tensor, ...]:
    #     if audio_input["type"] == "audio_embeds":
    #         audio_embeds = audio_input["audio_embeds"]
    #         return tuple(audio_embeds)

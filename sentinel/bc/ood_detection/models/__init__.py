from typing import Type

from .embedding_models import EmbeddingModel, ResnetEmbeddingModel, ClipEmbeddingModel
from .vlm_models import (
    VisionLanguageModel,
    OpenAIVisionLanguage,
    AnthropicVisionLanguage,
    GoogleVisionLanguage,
)


def get_embedding_cls(
    model: str,
) -> Type[EmbeddingModel]:
    if model == "resnet":
        return ResnetEmbeddingModel
    elif model == "clip":
        ClipEmbeddingModel
    else:
        raise ValueError(f"Model {model} is not supported.")


def get_vlm_cls(
    model: str,
) -> Type[VisionLanguageModel]:
    if "gpt" in model:
        return OpenAIVisionLanguage
    elif "claude" in model:
        return AnthropicVisionLanguage
    elif "gemini" in model:
        return GoogleVisionLanguage
    else:
        raise ValueError(f"Model {model} is not supported.")

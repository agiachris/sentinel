from typing import Optional, Union, Dict, Any, Iterable, Tuple, Any, List

import os
import re
import time
from abc import ABC, abstractmethod
from openai import OpenAI
from openai.types import chat as openai_types
from anthropic import Anthropic, types as anthropic_types
import google.generativeai as googleai
from google.generativeai import types as google_types, GenerativeModel


# Set the logging level for httpx to WARNING or ERROR to suppress INFO messages.
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)


from sentinel.bc.ood_detection.models.utils import (
    MessagesType,
    MODEL_TO_CORP,
    MODEL_TO_ID,
    delete_temp_file,
)


class VisionLanguageModel(ABC):

    def __init__(
        self,
        model: str,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ):
        """Construct VisionLanguageModel"""
        if api_key is None:
            api_key = os.environ.get(f"{MODEL_TO_CORP[model]}_API_KEY")
            if api_key is None:
                raise ValueError("API key must be provided.")

        self._api_key = api_key
        self._model = MODEL_TO_ID.get(model, model)
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = self.create_client()

    @abstractmethod
    def create_client(self, **kwargs: Any) -> Union[OpenAI, Anthropic, GenerativeModel]:
        """Create client."""
        raise NotImplementedError

    @staticmethod
    def process_response(response: str) -> Tuple[bool, int, str]:
        """Process response."""
        match: Union[re.Match | None] = re.search(
            r"Overall assessment:\s*(\w+)", response
        )
        if match is not None:
            status = 0
            assessment = match.group(1)
            pred = False if assessment == "ok" else True
            if assessment not in {"ok", "failure"}:
                status = 1
        else:
            status = 1
            pred = True

        return pred, status, response

    @abstractmethod
    def forward(
        self,
        messages: MessagesType,
        client_kwargs: Dict[str, Any],
    ) -> Tuple[bool, int, str]:
        """Forward pass."""
        raise NotImplementedError


class OpenAIVisionLanguage(VisionLanguageModel):

    def create_client(self, **kwargs: Any) -> OpenAI:
        return OpenAI(api_key=self._api_key, **kwargs)

    def forward(
        self,
        messages: Iterable[openai_types.ChatCompletionMessageParam],
        client_kwargs: Dict[str, Any],
    ) -> Tuple[bool, int, str]:
        success = False
        while not success:
            try:
                response: openai_types.ChatCompletion = (
                    self._client.chat.completions.create(
                        model=self._model,
                        max_tokens=self._max_tokens,
                        temperature=self._temperature,
                        messages=messages,
                        **client_kwargs,
                    )
                )
                success = True
            except:
                time.sleep(10)
                continue

        response_str = response.choices[0].message.content
        return VisionLanguageModel.process_response(response_str)


class AnthropicVisionLanguage(VisionLanguageModel):

    def create_client(self, **kwargs: Any) -> Anthropic:
        return Anthropic(api_key=self._api_key, **kwargs)

    def forward(
        self,
        messages: Iterable[anthropic_types.MessageParam],
        client_kwargs: Dict[str, Any],
    ) -> Tuple[bool, int, str]:
        success = False
        while not success:
            try:
                response: anthropic_types.Message = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    messages=messages,
                    **client_kwargs,
                )
                success = True
            except:
                time.sleep(10)
                continue

        response_str = response.content[0].text
        return VisionLanguageModel.process_response(response_str)


class GoogleVisionLanguage(VisionLanguageModel):

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        googleai.configure(api_key=self._api_key)

    def create_client(self, **kwargs: Any) -> GenerativeModel:
        return GenerativeModel(model_name=self._model, **kwargs)

    def forward(
        self,
        messages: google_types.ContentsType,
        client_kwargs: Dict[str, Any],
    ) -> Tuple[bool, int, str]:
        # Client must be constructed with system prompt.
        self._client = self.create_client(**client_kwargs)

        # Upload files.
        _tempfiles: List[str] = []
        _gfiles: List[google_types.File] = []
        _messages = []
        for message in messages:
            assert isinstance(message, str)
            if ".jpg" in messages or ".mp4" in message:
                gfile = googleai.upload_file(path=message, display_name=message)

                # Verify that .jpg or .mp4 was successfully uploaded.
                while gfile.state.name == "PROCESSING":
                    time.sleep(10)
                    gfile = googleai.get_file(gfile.name)
                if gfile.state.name == "FAILED":
                    raise ValueError(gfile.state.name)

                _messages.append(gfile)
                _tempfiles.append(message)
                _gfiles.append(gfile)

            else:
                _messages.append(message)

        success = False
        while not success:
            try:
                response: google_types.GenerateContentResponse = (
                    self._client.generate_content(
                        _messages,
                        generation_config=google_types.GenerationConfig(
                            max_output_tokens=self._max_tokens,
                            temperature=self._temperature,
                        ),
                    )
                )
                success = True
            except:
                time.sleep(10)
                continue

        # Clear files.
        for f in _tempfiles:
            delete_temp_file(f)
        for f in _gfiles:
            googleai.delete_file(f.name)
        # Does not support multiple parallel processes.
        # for f in googleai.list_files():
        #     googleai.delete_file(f.name)

        response_str = response.text
        return VisionLanguageModel.process_response(response_str)

from typing import Any, Dict, Union, List, Optional, Iterable, Tuple

import os
import yaml
import imageio
import pathlib
import numpy as np
from copy import deepcopy
from openai.types import chat as openai_types
from anthropic import types as anthropic_types
from google.generativeai import types as google_types

from sentinel.bc.ood_detection.models.utils import (
    resize_image,
    center_crop_image,
    image_to_bytes,
    create_temp_jpg,
    create_temp_video,
    MessagesType,
    CORP_TO_MODEL,
)


DEBUG = 0
BREAKPOINT = 0
dbprint = print if DEBUG == 1 else lambda *args: ()


def create_gif(
    images: List[np.ndarray],
    filename: str,
) -> None:
    filepath = pathlib.Path(os.path.dirname(__file__)) / f"{filename}.gif"
    imageio.mimsave(filepath, images, fps=1, loop=0)


CROP_DOMAIN_PARAMS = {
    "push_chair": {
        "center_y_ratio": None,
        "center_x_ratio": 0.6,
        "crop_y_ratio": None,
        "crop_x_ratio": 1 / 1.9,
    },
}

RESIZE_DOMAIN_PARAMS = {
    "gpt-4-turbo": {},
    "gpt-4o": {},
    "claude-3-5-sonnet-20240620": {},
    "gemini-1-5-pro": {},
}


class PromptManager:

    def __init__(
        self,
        template: Union[str, pathlib.Path],
        domain: str,
        crop_images: bool = False,
        resize_images: bool = False,
        subsample_freq: int = 1,
    ):
        with open(template, "r") as f:
            self._template: Dict[str, Any] = yaml.safe_load(f)
        self._domain = domain
        self._crop_images = crop_images
        self._resize_images = resize_images
        self._subsample_freq = subsample_freq

        # Extract domain prompts.
        self._prompts = self._template["prompts"][self._domain]
        assert isinstance(self._prompts, list) and len(self._prompts) <= 2

        # Dynamically set.
        self._reference = None
        self._goals = None
        self._time = None

    @property
    def settings(self) -> Dict[str, Any]:
        """Return prompt settings."""
        return self._template["settings"]

    @property
    def description(self) -> str:
        """Return task description."""
        assert "descriptions" in self._template.keys()
        return self._template["descriptions"][self._domain]

    @property
    def time(self) -> Optional[int]:
        """Return task time."""
        return self._time if self.settings.get("use_time", False) else None

    @time.setter
    def time(self, value: int) -> None:
        """Set task time."""
        self._time = value

    @property
    def reference(self) -> Optional[List[np.ndarray]]:
        """Return reference video."""
        return self._reference if self.settings.get("use_reference", False) else None

    @reference.setter
    def reference(self, value: List[np.ndarray]) -> None:
        """Set reference video."""
        assert all(isinstance(x, np.ndarray) and x.ndim == 3 for x in value)
        self._reference = deepcopy(value)

    @property
    def reference_episode(self) -> Optional[int]:
        """Get reference video index."""
        return (
            self._template["references"][self._domain]
            if "references" in self._template.keys()
            else None
        )

    @property
    def goals(self) -> Optional[List[np.ndarray]]:
        """Return goal images."""
        return self._goals if self.settings.get("use_goals", False) else None

    @goals.setter
    def goals(self, value: List[np.ndarray]) -> None:
        """Set goal images."""
        assert all(isinstance(x, np.ndarray) and x.ndim == 3 for x in value)
        self._goals = deepcopy(value)

    @property
    def goal_episodes(self) -> Optional[List[int]]:
        """Get reference video index."""
        return (
            self._template["goals"][self._domain]
            if "goals" in self._template.keys()
            else None
        )

    @property
    def time_limit(self) -> Optional[int]:
        """Return task time limit."""
        return (
            int(
                self._template["time_limits"][self._domain] / self._subsample_freq + 0.5
            )
            if "time_limits" in self._template.keys()
            else None
        )

    @property
    def questions(self) -> Optional[str]:
        """Return task questions."""
        return (
            self._template["questions"][self._domain].strip()
            if "questions" in self._template.keys()
            else None
        )

    def process_str_prompt(self, prompt: str) -> str:
        """Process string prompt."""
        prompt = prompt.strip().replace("{DESCRIPTION}", self.description)

        if self.settings.get("use_time", False):
            assert isinstance(self.time, int)
            prompt = prompt.replace("{TIME}", str(self.time))

        if self.settings.get("use_time_limit", False):
            assert isinstance(self.time_limit, int)
            prompt = prompt.replace("{TIME_LIMIT}", str(self.time_limit))
            if self.settings.get("use_time", False):
                assert self.time <= self.time_limit

        if self.settings.get("use_questions", False):
            assert isinstance(self.questions, str)
            prompt = prompt.replace("{QUESTIONS}", self.questions)

        return prompt

    def process_video_prompt(
        self, model: str, images: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Process video prompt."""
        if self._domain in RESIZE_DOMAIN_PARAMS[model]:
            resize_params = RESIZE_DOMAIN_PARAMS[model][self._domain]
            images = [resize_image(image, **resize_params) for image in images]

        if self._domain in CROP_DOMAIN_PARAMS:
            crop_kwargs = CROP_DOMAIN_PARAMS[self._domain]
            images = [center_crop_image(image, **crop_kwargs) for image in images]

        if DEBUG:
            create_gif(images, "cropped_video")
        if BREAKPOINT:
            breakpoint()

        return images

    def construct_openai_prompt(
        self, system_prompt: Optional[str], user_prompt: List[Dict[str, Any]]
    ) -> Tuple[Iterable[openai_types.ChatCompletionMessageParam], Dict[str, Any]]:
        """Construct OpenAI prompt."""
        client_kwargs = {}

        messages = []
        if isinstance(system_prompt, str):
            messages.append({"role": "system", "content": system_prompt})

        user_content = []
        for prompt in user_prompt:
            if prompt["type"] == "text":
                user_content.append(prompt)

            elif prompt["type"] in {"image", "video"}:
                base64_images = [image_to_bytes(image) for image in prompt["data"]]
                user_content += list(
                    map(
                        lambda x: {
                            "type": "image_url",
                            "image_url": {
                                "detail": "low",
                                "url": f"data:image/jpeg;base64,{x}",
                            },
                        },
                        base64_images,
                    )
                )

        messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

        return messages, client_kwargs

    def construct_anthropic_prompt(
        self, system_prompt: Optional[str], user_prompt: List[Dict[str, Any]]
    ) -> Tuple[Iterable[anthropic_types.MessageParam], Dict[str, Any]]:
        """Construct Anthropic prompt."""
        client_kwargs = (
            {"system": system_prompt} if isinstance(system_prompt, str) else {}
        )

        user_content = []
        for prompt in user_prompt:
            if prompt["type"] == "text":
                user_content.append(prompt)

            elif prompt["type"] in {"image", "video"}:
                base64_images = [image_to_bytes(image) for image in prompt["data"]]
                user_content += list(
                    map(
                        lambda x: {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": x,
                            },
                        },
                        base64_images,
                    )
                )

        messages = [{"role": "user", "content": user_content}]

        return messages, client_kwargs

    def construct_google_prompt(
        self, system_prompt: Optional[str], user_prompt: List[Dict[str, Any]]
    ) -> Tuple[google_types.ContentsType, Dict[str, Any]]:
        """Construct Google prompt."""
        client_kwargs = (
            {"system_instruction": system_prompt}
            if isinstance(system_prompt, str)
            else {}
        )

        messages = []
        for prompt in user_prompt:
            if prompt["type"] == "text":
                messages.append(prompt["text"])

            elif prompt["type"] == "image":
                assert len(prompt["data"]) == 1
                image_file = create_temp_jpg(prompt["data"][0])
                messages.append(image_file)

            elif prompt["type"] == "video":
                video_file = create_temp_video(prompt["data"], fps=1)
                messages.append(video_file)

        return messages, client_kwargs

    def construct_prompt(
        self,
        model: str,
        images: List[np.ndarray],
    ) -> Tuple[MessagesType, Dict[str, Any]]:
        """Construct model prompt."""
        self.time = len(images)

        # Get system prompt.
        prompts = deepcopy(self._prompts)
        system_prompt = (
            self.process_str_prompt(prompts[0]["content"])
            if prompts[0]["role"] == "system"
            else None
        )

        # Get user prompts.
        user_prompt = prompts[1] if isinstance(system_prompt, str) else prompts[0]
        assert user_prompt["role"] == "user" and isinstance(
            user_prompt["content"], list
        )

        _user_prompt = []
        for prompt in user_prompt["content"]:
            if prompt["type"] == "text":
                _user_prompt.append(
                    {"type": "text", "text": self.process_str_prompt(prompt["text"])}
                )

            elif prompt["type"] == "image":
                assert (
                    not self.settings.get("use_video", False)
                    and prompt["image"] == "{CURRENT}"
                )
                _user_prompt.append(
                    {
                        "type": "image",
                        "data": self.process_video_prompt(model, deepcopy(images[-1:])),
                    }
                )

            elif prompt["type"] == "goals":
                assert (
                    self.settings.get("use_goals", False)
                    and prompt["goals"] == "{GOALS}"
                )
                _user_prompt.append(
                    {
                        "type": "video",
                        "data": self.process_video_prompt(model, deepcopy(self.goals)),
                    }
                )

            elif prompt["type"] == "video":
                assert self.settings.get("use_video", False) and prompt["video"] in {
                    "{CURRENT}",
                    "{REFERENCE}",
                }
                video = images if prompt["video"] == "{CURRENT}" else self.reference
                _user_prompt.append(
                    {
                        "type": "video",
                        "data": self.process_video_prompt(model, deepcopy(video)),
                    }
                )

            else:
                raise ValueError(f"Prompt type {prompt['type']} is not supported.")

        if model in CORP_TO_MODEL["OPENAI"]:
            messages, client_kwargs = self.construct_openai_prompt(
                system_prompt, _user_prompt
            )
        elif model in CORP_TO_MODEL["ANTHROPIC"]:
            messages, client_kwargs = self.construct_anthropic_prompt(
                system_prompt, _user_prompt
            )
        elif model in CORP_TO_MODEL["GOOGLE"]:
            messages, client_kwargs = self.construct_google_prompt(
                system_prompt, _user_prompt
            )
        else:
            raise ValueError(f"Model {model} is not supported.")

        return messages, client_kwargs

from typing import TypeVar, Iterable, Optional, List

import os
import cv2
import base64
import tempfile
import numpy as np
from PIL import Image
from io import BytesIO
from openai.types import chat as openai_types
from anthropic import types as anthropic_types
from google.generativeai import types as google_types


MessagesType = TypeVar(
    "MessagesType",
    Iterable[openai_types.ChatCompletionMessageParam],
    Iterable[anthropic_types.MessageParam],
    google_types.ContentsType,
)

ResponseType = TypeVar(
    "ResponseType",
    openai_types.ChatCompletion,
    anthropic_types.Message,
    google_types.GenerateContentResponse,
)

MODEL_TO_CORP = {
    "gpt-4-turbo": "OPENAI",
    "gpt-4o": "OPENAI",
    "claude-3-5-sonnet-20240620": "ANTHROPIC",
    "gemini-1-5-pro": "GOOGLE",
}

CORP_TO_MODEL = {
    "OPENAI": {"gpt-4o", "gpt-4-turbo"},
    "ANTHROPIC": {"claude-3-5-sonnet-20240620"},
    "GOOGLE": {"gemini-1-5-pro"},
}

MODEL_TO_ID = {"gemini-1-5-pro": "gemini-1.5-pro"}


def center_crop_image(
    image: np.ndarray,
    center_y_ratio: Optional[int] = None,
    center_x_ratio: Optional[int] = None,
    crop_y_ratio: Optional[int] = None,
    crop_x_ratio: Optional[int] = None,
) -> np.ndarray:
    """Crop the image to a specified size around a center pixel."""
    height, width = image.shape[:2]

    # Crop height and width.
    center_y = int(
        height * center_y_ratio if center_y_ratio is not None else height // 2
    )
    center_x = int(width * center_x_ratio if center_x_ratio is not None else width // 2)
    crop_height = int(
        (height * crop_y_ratio if crop_y_ratio is not None else height) // 2
    )
    crop_width = int((width * crop_x_ratio if crop_x_ratio is not None else width) // 2)

    # Crop image.
    start_y = max(center_y - crop_height, 0)
    end_y = min(center_y + crop_height, height)
    start_x = max(center_x - crop_width, 0)
    end_x = min(center_x + crop_width, width)
    cropped_image = image[start_y:end_y, start_x:end_x]

    return cropped_image


def crop_to_square(
    image: np.ndarray,
    center_y_ratio: Optional[float] = None,
    center_x_ratio: Optional[float] = None,
    side_ratio: Optional[float] = None,
) -> np.ndarray:
    """Crop the image to a square of specified size around a center pixel."""
    height, width = image.shape[:2]

    # Set default values for center and side length if not provided.
    center_y = int(
        height * center_y_ratio if center_y_ratio is not None else height // 2
    )
    center_x = int(width * center_x_ratio if center_x_ratio is not None else width // 2)
    side_length = int(
        width * side_ratio if side_ratio is not None else min(height, width)
    )

    # Half the side length to determine cropping boundaries.
    half_side = side_length // 2

    # Calculate start and end coordinates for cropping.
    start_y = max(center_y - half_side, 0)
    end_y = min(center_y + half_side, height)
    start_x = max(center_x - half_side, 0)
    end_x = min(center_x + half_side, width)

    # Crop the image to a square.
    cropped_image = image[start_y:end_y, start_x:end_x]

    return cropped_image


def resize_image(
    image: np.ndarray,
    scale: Optional[float] = None,
    y: Optional[int] = None,
    x: Optional[int] = None,
) -> None:
    """Resize image to specified height (y) and width (x)."""
    if scale is not None:
        height, width = image.shape[:2]
        y, x = int(height * scale), int(width * scale)
    return cv2.resize(image, (x, y))


def image_to_bytes(image: np.ndarray) -> str:
    """Convert image to bytes."""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


def bytes_to_pil(image: str) -> Image:
    """Convert bytes to image."""
    return Image.open(BytesIO(base64.b64decode(image.encode("utf-8"))))


def create_temp_video(images: List[np.ndarray], fps: int = 1) -> str:
    """Create a temporary .mp4 video file from a list of images."""
    height, width = images[0].shape[:2]

    # Create a temporary file.
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_video_path = temp_video.name
    temp_video.close()

    # Initialize video writer.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    # Write each image to the video.
    for img in images:
        video_writer.write(img)

    # Release the video writer.
    video_writer.release()

    return temp_video_path


def create_temp_jpg(image: np.ndarray) -> str:
    """Create a temporary .jpg file from a NumPy array."""
    # Create a temporary file with a .jpg suffix.
    temp_jpg = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    temp_jpg_path = temp_jpg.name
    temp_jpg.close()

    # Write the NumPy array as a .jpg image.
    cv2.imwrite(temp_jpg_path, image)

    return temp_jpg_path


def delete_temp_file(file_path: str) -> None:
    """Delete the temporary file after use."""
    if os.path.exists(file_path):
        os.remove(file_path)

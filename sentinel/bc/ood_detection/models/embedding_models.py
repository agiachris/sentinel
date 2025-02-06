import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from torchvision.models import resnet18, ResNet18_Weights
from abc import ABC, abstractmethod


class EmbeddingModel(ABC):

    def __init__(self, device="cpu"):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__()

    @abstractmethod
    def get_embedding(self, image):
        pass


class ResnetEmbeddingModel(EmbeddingModel):

    def __init__(self, device="cpu"):
        super().__init__(device=device)

        weights = ResNet18_Weights.DEFAULT
        self.transform = weights.transforms()

        self.base_model = resnet18(weights=weights)
        self.embedding_model = nn.Sequential(*list(self.base_model.children())[:-1]).to(
            self.device
        )

    @torch.no_grad()
    def get_embedding(self, image):
        image = torch.Tensor(image).permute(2, 0, 1)
        input = self.transform(image).to(self.device)
        embedding = self.embedding_model(input.unsqueeze(0)).flatten().cpu().numpy()
        return embedding


class ClipEmbeddingModel(EmbeddingModel):

    def __init__(self, device="cpu", clip_model="openai/clip-vit-base-patch32"):
        super().__init__(device=device)
        self.base_model = CLIPModel.from_pretrained(clip_model)
        self.processor = CLIPProcessor.from_pretrained(clip_model)
        self.embedding_model = self.base_model.vision_model.to(self.device)

    @torch.no_grad()
    def get_embedding(self, image):
        input = self.processor(images=[image], return_tensors="pt").to(self.device)
        output = self.embedding_model(**input)
        embedding = output.pooler_output.cpu().flatten().numpy()
        return embedding

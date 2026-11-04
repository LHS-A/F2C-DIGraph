"""Image encoders for CFP and three-channel grayscale FFA inputs."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torchvision import models


class ResNet50Encoder(nn.Module):
    """ImageNet-pretrained ResNet-50 with dense tokens and global features."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 768, pretrained: bool = True) -> None:
        super().__init__()
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            resnet = models.resnet50(weights=weights)
        except (AttributeError, TypeError):
            # torchvision<0.13 uses the legacy ``pretrained`` argument.
            resnet = models.resnet50(pretrained=pretrained)
        if in_channels != 3:
            old = resnet.conv1
            new = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
            with torch.no_grad():
                mean_weight = old.weight.mean(dim=1, keepdim=True)
                new.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))
            resnet.conv1 = new
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj_global = nn.Linear(2048, embed_dim)
        self.proj_tokens = nn.Conv2d(2048, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.stem(x)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        fmap = self.layer4(z)
        pooled = self.pool(fmap).flatten(1)
        global_feat = self.proj_global(pooled)
        tokens = self.proj_tokens(fmap).flatten(2).transpose(1, 2)
        return {"global": global_feat, "tokens": tokens, "feature_map": fmap}


class ImageClassifier(nn.Module):
    """Backbone plus linear disease classifier."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embed_dim: int = 768,
        pretrained: bool = True,
        backbone: str = "resnet50",
    ) -> None:
        super().__init__()
        if backbone != "resnet50":
            raise ValueError("Strict reproduction uses ImageNet-pretrained ResNet-50 only.")
        self.encoder = ResNet50Encoder(in_channels=in_channels, embed_dim=embed_dim, pretrained=pretrained)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.encoder(x)
        out["logits"] = self.classifier(out["global"])
        return out

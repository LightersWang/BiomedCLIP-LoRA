import open_clip
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer as timm_ViT

from .lora import LoRA_ViT_timm


class BiomedCLIPViT_LoRA(nn.Module):
    MODEL_TAG = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'

    def __init__(self, lora_rank=4):
        super().__init__()
        self.lora_rank = lora_rank
        biomedclip = open_clip.create_model(self.MODEL_TAG)

        # LoRA-tune the vision transformer
        vit = biomedclip.visual.trunk
        assert isinstance(vit, timm_ViT)
        self.lora_vit = LoRA_ViT_timm(vit_model=vit, r=lora_rank)

    # get features from the vision transformer
    def forward(self, image):
        B = image.shape[0]
        # remove [CLS] token
        feat = self.lora_vit.lora_vit.forward_features(image)[:, 1:]  # [B, 196, 768]
        feat = feat.reshape(B, -1, 14, 14)  # [B, 768, 14, 14]
        return feat


def biomedclip_lora(lora_rank):
    return BiomedCLIPViT_LoRA(lora_rank=lora_rank)
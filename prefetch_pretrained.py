"""Download the ImageNet / V-JEPA 2 initialisations the model constructors ask for, so the container
runs offline. (Our fine-tuned weights in weights/ overwrite them.)"""
import timm
from transformers import AutoModel
AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-256")
for n in ["resnet50.a1_in1k", "convnext_small.fb_in22k_ft_in1k"]:
    timm.create_model(n, pretrained=True)
from transformers import VitPoseForPoseEstimation
VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-base")
AutoModel.from_pretrained("apiantonio/vjepa2.1-vit-base-384", revision="ea7765861aa689985c593727725afa378fc87492",
                          trust_remote_code=True)          # V-JEPA 2.1 ViT-B (Meta weights, HF port; code read and pinned)
AutoModel.from_pretrained("apiantonio/vjepa2.1-vit-large-384", revision="d6cfdbdd818754f22eaa72e5320d97724765f099",
                          trust_remote_code=True)          # V-JEPA 2.1 ViT-L (same port, pinned)
print("pretrained weights cached")

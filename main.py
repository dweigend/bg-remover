import os

os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
from transformers import AutoModelForImageSegmentation

# Load BiRefNet with weights
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to(device)
birefnet.eval()

print("Hello from BiRefNet!")

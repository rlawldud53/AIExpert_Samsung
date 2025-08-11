import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import joblib
from pathlib import Path

import clip
from tqdm.auto import tqdm

class ClipEmbed:
    def __init__(self, device):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model = self.model.eval()
        self.device = device

    def embed(self, text):
        with torch.inference_mode():
            text = clip.tokenize(text).to(self.device)
            text_emb = self.model.encode_text(text)[0].cpu()
        return text_emb


def get_transform(image_size, normalize=False):
    ops = [T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()]
    if normalize: 
        ops += [T.Normalize((0.5,)*3, (0.5,)*3)]
    return T.Compose(ops)

class CelebADialogDataset(Dataset):
    def __init__(self, img_dir, ann_jsonl, image_size=32, normalize=False, clip_embedder=None):
        assert clip_embedder is not None, "Pass a ClipEmbed instance"
        self.img_dir = Path(img_dir)
        with open(ann_jsonl, "r", encoding="utf-8") as f:
            self.items = [json.loads(l) for l in f if l.strip()]
        self.transform = get_transform(image_size, normalize)
        self.clip = clip_embedder
        self.device = getattr(self.clip, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def __len__(self): 
        return len(self.items)

    @torch.no_grad()
    def _encode(self, text: str):
        z = self.clip.encode([text]).to(self.device)        # (1, D)  e.g., D=512
        return z.squeeze(0).detach().cpu()                  # (D,)

    def __getitem__(self, idx):
        rec = self.items[idx]; name, text = rec["image"], rec["caption"]
        img = self.transform(Image.open(self.img_dir/name).convert("RGB"))
        emb = self._encode(text)
        return img, {"text_emb": emb, "text": text}

    @torch.no_grad()
    def embed_new_text(self, text: str, clip_embedder=None):
        ce = clip_embedder or self.clip
        z = ce.encode([text]).to(self.device)
        return z.squeeze(0).detach().cpu()


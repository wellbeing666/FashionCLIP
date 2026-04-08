from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # 如果没有 tqdm，就退化为普通迭代


class ClipEmbedder:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        use_fp16: bool = False,
    ) -> None:
        """
        Args:
            model_name: HuggingFace 模型名称，如 "openai/clip-vit-base-patch32" 或 "patrickjohncyh/fashion-clip"
            device: 运行设备 ("cuda" 或 "cpu")，若为 None 则自动检测
            use_fp16: 是否使用半精度（仅当 device 为 cuda 时有效，可加速并省显存）
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.use_fp16 = use_fp16 and self.device == "cuda"
        print("Using device:", self.device)

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        if self.use_fp16:
            self.model = self.model.half()  # 转为半精度

        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_images(
        self,
        image_paths: List[str],
        batch_size: int = 16,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        编码一批图片，返回归一化后的嵌入向量 (N, dim)
        如果某张图片无法打开，会跳过并打印警告，最终返回的数组长度可能小于输入列表。
        """
        valid_paths = []
        valid_images = []

        # 预加载有效的图片
        it = tqdm(image_paths, desc="Loading images", disable=not show_progress)
        for p in it:
            try:
                img = Image.open(p).convert("RGB")
                valid_paths.append(p)
                valid_images.append(img)
            except Exception as e:
                print(f"Warning: cannot load image {p}: {e}")

        if not valid_images:
            return np.empty((0, self.model.config.projection_dim))

        all_embeddings = []
        num_batches = (len(valid_images) + batch_size - 1) // batch_size
        batch_iter = tqdm(range(num_batches), desc="Encoding images", disable=not show_progress)

        for batch_idx in batch_iter:
            start = batch_idx * batch_size
            end = start + batch_size
            batch_imgs = valid_images[start:end]

            inputs = self.processor(
                images=batch_imgs, return_tensors="pt", padding=True
            ).to(self.device)

            # 如果使用半精度，输入也要转为 half
            if self.use_fp16:
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

            outputs = self.model.get_image_features(**inputs)

            # 兼容不同 transformers 版本
            if isinstance(outputs, torch.Tensor):
                features = outputs
            else:
                features = outputs.image_embeds

            features = torch.nn.functional.normalize(features, p=2, dim=1)

            all_embeddings.append(features.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """编码文本列表，返回归一化嵌入向量"""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        if self.use_fp16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
        outputs = self.model(**inputs)
        features = outputs.text_embeds
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().numpy()


def save_embeddings(
    path: str,
    embeddings: np.ndarray,
    image_paths: Optional[List[str]] = None,
) -> None:
    """
    保存嵌入向量，并可选的保存对应的图片路径列表（用于顺序校验）
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, embeddings)

    if image_paths is not None:
        # 将图片路径保存为 .npy 或 .txt
        paths_path = out.with_suffix(".paths.npy")
        np.save(paths_path, np.array(image_paths, dtype=object))


def load_embeddings(path: str) -> np.ndarray:
    return np.load(path)


def load_embedding_paths(path: str) -> List[str]:
    """加载之前保存的图片路径列表"""
    paths_path = Path(path).with_suffix(".paths.npy")
    if paths_path.exists():
        return np.load(paths_path, allow_pickle=True).tolist()
    return []
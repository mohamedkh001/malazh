"""Utilities for loading a FaceNet model and generating normalized embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image


DEFAULT_IMAGE_SIZE = 160
DEFAULT_EMBEDDING_SIZE = 512


def build_model(pretrained: bool = True) -> InceptionResnetV1:
    """Build an InceptionResnetV1 model configured for embedding extraction."""
    model = InceptionResnetV1(pretrained="vggface2" if pretrained else None)
    model.eval()
    return model


def save_model_weights(model_path: Path, pretrained: bool = True) -> Path:
    """Create and save FaceNet model weights to a .pth file."""
    model = build_model(pretrained=pretrained)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    return model_path


def load_model(model_path: Path, device: Optional[str] = None) -> InceptionResnetV1:
    """Load model weights from a .pth file."""
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False)
    state_dict = torch.load(model_path, map_location=resolved_device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(resolved_device)
    return model


def preprocess_image(image_path: Path, image_size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    """Load, resize, and normalize a face image into FaceNet input format."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    image_array = np.asarray(image, dtype=np.float32)

    image_array = (image_array - 127.5) / 128.0
    image_array = np.transpose(image_array, (2, 0, 1))

    tensor = torch.from_numpy(image_array).unsqueeze(0)
    return tensor


def extract_embedding(model: InceptionResnetV1, image_path: Path, device: Optional[str] = None) -> np.ndarray:
    """Generate an L2-normalized embedding from a single image."""
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = preprocess_image(image_path).to(resolved_device)

    with torch.no_grad():
        embedding = model(image_tensor).cpu().numpy()[0]

    norm = np.linalg.norm(embedding)
    if norm == 0:
        raise ValueError(f"Zero-vector embedding produced for image: {image_path}")

    return embedding / norm


def cli() -> None:
    parser = argparse.ArgumentParser(description="Extract a single normalized face embedding.")
    parser.add_argument("--model", type=Path, default=Path("model.pth"), help="Path to .pth model")
    parser.add_argument("--image", type=Path, required=False, help="Path to image")
    parser.add_argument("--init-model", action="store_true", help="Create model.pth from pretrained FaceNet")
    args = parser.parse_args()

    if args.init_model:
        save_model_weights(args.model)
        print(f"Saved model weights to: {args.model}")

    if args.image:
        model = load_model(args.model)
        emb = extract_embedding(model, args.image)
        print(f"Embedding shape: {emb.shape}")
        print(np.array2string(emb, precision=5, suppress_small=False, max_line_width=120))


if __name__ == "__main__":
    cli()

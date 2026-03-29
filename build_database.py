"""Create embeddings.npy from a Dataset directory structure.

Expected structure:
Dataset/
  person_a/
    img1.jpg
    img2.jpg
  person_b/
    img1.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from embedding import DEFAULT_EMBEDDING_SIZE, load_model, extract_embedding

VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"}


def list_images(dataset_dir: Path) -> list[tuple[str, Path]]:
    samples: list[tuple[str, Path]] = []
    for person_dir in sorted(dataset_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        person_name = person_dir.name
        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() in VALID_SUFFIXES:
                samples.append((person_name, image_path))
    return samples


def build_database(model_path: Path, dataset_dir: Path, output_path: Path) -> Path:
    model = load_model(model_path)
    samples = list_images(dataset_dir)
    if not samples:
        raise ValueError(f"No valid images found under: {dataset_dir}")

    labels: list[str] = []
    paths: list[str] = []
    embeddings: list[np.ndarray] = []

    for label, image_path in samples:
        embedding = extract_embedding(model, image_path)
        if embedding.shape[0] != DEFAULT_EMBEDDING_SIZE:
            raise ValueError(f"Unexpected embedding shape {embedding.shape} for {image_path}")
        labels.append(label)
        paths.append(str(image_path.as_posix()))
        embeddings.append(embedding.astype(np.float32))
        print(f"Processed {image_path}")

    db = {
        "labels": np.array(labels),
        "paths": np.array(paths),
        "embeddings": np.stack(embeddings, axis=0),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, db, allow_pickle=True)
    return output_path


def cli() -> None:
    parser = argparse.ArgumentParser(description="Build face embedding database from Dataset folder.")
    parser.add_argument("--model", type=Path, default=Path("model.pth"))
    parser.add_argument("--dataset", type=Path, default=Path("Dataset"))
    parser.add_argument("--output", type=Path, default=Path("embeddings.npy"))
    args = parser.parse_args()

    output = build_database(args.model, args.dataset, args.output)
    print(f"Saved embedding database to {output}")


if __name__ == "__main__":
    cli()

"""Search for the closest face match in embeddings.npy using cosine similarity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from embedding import load_model, extract_embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search_face(model_path: Path, database_path: Path, query_image: Path) -> dict[str, object]:
    model = load_model(model_path)
    query_embedding = extract_embedding(model, query_image)

    db = np.load(database_path, allow_pickle=True).item()
    labels = db["labels"]
    paths = db["paths"]
    embeddings = db["embeddings"]

    similarities = np.dot(embeddings, query_embedding)
    best_idx = int(np.argmax(similarities))
    best_similarity = float(similarities[best_idx])

    return {
        "label": str(labels[best_idx]),
        "image_path": str(paths[best_idx]),
        "similarity": best_similarity,
    }


def cli() -> None:
    parser = argparse.ArgumentParser(description="Search nearest face match.")
    parser.add_argument("--model", type=Path, default=Path("model.pth"))
    parser.add_argument("--database", type=Path, default=Path("embeddings.npy"))
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()

    result = search_face(args.model, args.database, args.image)
    print("Best match:")
    print(f"  Label      : {result['label']}")
    print(f"  Image path : {result['image_path']}")
    print(f"  Similarity : {result['similarity']:.6f}")


if __name__ == "__main__":
    cli()

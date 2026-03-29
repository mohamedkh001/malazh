# Face Recognition (PyTorch + TFLite) for Flutter Offline Integration

This project provides an end-to-end pipeline to:

1. Build/load a FaceNet model in PyTorch (`model.pth`)
2. Extract normalized face embeddings
3. Create an embeddings database (`embeddings.npy`) from `Dataset/`
4. Search the nearest match using cosine similarity
5. Convert the PyTorch model to TensorFlow Lite (`model.tflite`) for Flutter offline usage

## Project Structure

```text
project/
├─ Dataset/
│  ├─ alice/
│  ├─ bob/
│  └─ charlie/
├─ embeddings.npy
├─ model.pth
├─ model.tflite
├─ embedding.py
├─ build_database.py
├─ search.py
├─ convert_to_tflite.py
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

## 1) Setup (Local)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Create `model.pth` (Pretrained FaceNet Weights)

```bash
python embedding.py --init-model --model model.pth
```

This saves an InceptionResnetV1 model state dictionary as `model.pth`.

## 3) Build Embeddings Database

Place images in person-specific folders under `Dataset/`.

Example:

```text
Dataset/
├─ alice/
│  ├─ alice_1.png
│  ├─ alice_2.png
├─ bob/
│  ├─ bob_1.png
│  ├─ bob_2.png
└─ charlie/
   ├─ charlie_1.png
   ├─ charlie_2.png
```

Build database:

```bash
python build_database.py --model model.pth --dataset Dataset --output embeddings.npy
```

## 4) Search a New Image

```bash
python search.py --model model.pth --database embeddings.npy --image Dataset/alice/alice_1.png
```

Output includes:
- predicted person label
- matched image path
- cosine similarity score

## 5) Convert PyTorch `.pth` to TensorFlow Lite `.tflite`

```bash
python convert_to_tflite.py --model model.pth --output model.tflite
```

This script runs:
- PyTorch -> ONNX
- ONNX -> TensorFlow SavedModel
- SavedModel -> TFLite

## 6) Docker Usage

Build:

```bash
docker build -t face-recognition-offline .
```

Run shell:

```bash
docker run --rm -it -v "$PWD":/app face-recognition-offline
```

Inside container, execute scripts exactly as above.

## Flutter Handoff Notes

- Use `model.tflite` in Flutter (`tflite_flutter` plugin).
- Input tensor shape: `[1, 3, 160, 160]` (float32).
- Input normalization: `(pixel - 127.5) / 128.0`.
- Output embedding size: `512`.
- Always L2-normalize embeddings in Flutter before similarity comparison.

## Offline Considerations

- After generating `model.pth`, `model.tflite`, and `embeddings.npy`, inference is fully offline.
- The first model initialization may download pretrained weights once if not cached.

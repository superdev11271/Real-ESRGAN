# Real-ESRGAN ONNX

Real-ESRGAN image upscaling from exported ONNX models — as a Python class, a CLI, and a FastAPI server.

## Layout

| File | Purpose |
| --- | --- |
| [realesrgan_onnx.py](realesrgan_onnx.py) | `RealESRGANOnnx` — the upsampler (preprocess → session → postprocess) |
| [inference.py](inference.py) | CLI: upscale one image file |
| [server.py](server.py) | FastAPI server |
| `models/` | `.onnx` weights (not tracked in git) |

## Install

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `onnxruntime-gpu`. For a CPU-only machine, install `onnxruntime` instead.

## Library

```python
import cv2
from realesrgan_onnx import RealESRGANOnnx

upsampler = RealESRGANOnnx('models/RealESRGAN_x2plus.onnx', device='cuda')
output = upsampler.infer(cv2.imread('input.jpg'))          # BGR uint8 in, BGR uint8 out
outputs = upsampler.infer_batch([img1, img2])              # needs a model exported with --dynamic
```

- `device` is `'cuda'` (default) or `'cpu'`. `'cuda'` falls back to CPU when no CUDA execution provider is available.
- fp32 and fp16 models are both supported; the input dtype is read from the model.
- `infer_batch` requires all images to share one shape.

## CLI

```bash
python inference.py -i test/cat.jpg
python inference.py -i test/cat.jpg -o results -m models/RealESRGAN_x4plus.onnx -d cpu
```

| Flag | Default |
| --- | --- |
| `-i, --input` | required |
| `-o, --output` | `results` |
| `-m, --model_path` | `models/RealESRGAN_x2plus.onnx` |
| `-d, --device` | `cuda` |

Results are saved as `<name>_x<scale><ext>`, e.g. `cat_x2.jpg`.

## Server

```bash
python server.py
python server.py -m RealESRGAN_x4plus_fp16.onnx -d cpu -p 9000
```

| Flag | Default |
| --- | --- |
| `-m, --model` | `RealESRGAN_x2plus.onnx` (resolved inside `models/`) |
| `-d, --device` | `cuda` |
| `--max_side` | `1920` |
| `--host` | `0.0.0.0` |
| `-p, --port` | `8080` |

The model is loaded once at startup — a missing file fails immediately rather than on the first request.

An upload whose longer side exceeds `--max_side` is downscaled to that limit (keeping aspect ratio), upsampled, then resized back to its **original** dimensions — so oversized images come back the same size they went in, enhanced rather than enlarged. Images within the limit are upsampled normally and come back at the model's scale.

### `POST /api/upscale/`

Multipart upload, field name `image`. Returns the upscaled image as `image/png`.

```bash
curl -X POST -F "image=@test/cat.jpg" http://127.0.0.1:8080/api/upscale/ -o out.png
```

`400` if the upload cannot be decoded as an image.

### `GET /api/health/`

```bash
curl http://127.0.0.1:8080/api/health/
```

```json
{"status": "ok", "model": "RealESRGAN_x2plus.onnx", "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
```

`providers` comes from the live session, so it shows whether CUDA actually engaged or fell back to CPU. `503` if the model is not loaded.

Interactive docs at `/docs`.

## Docker

Base image `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04`. `models/` is not baked into the image (~500MB) — mount it.

```bash
docker build -t realesrgan-server .
docker run --gpus all -p 8080:8080 -v ./models:/app/models realesrgan-server
```

Server flags pass straight through the entrypoint:

```bash
docker run --gpus all -p 8080:8080 -v ./models:/app/models realesrgan-server -m RealESRGAN_x4plus_fp16.onnx
```

Omit `--gpus all` to run on CPU, and pass `-d cpu` to skip the CUDA probe.

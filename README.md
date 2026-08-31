# Real-ESRGAN ONNX

Real-ESRGAN image upscaling from exported ONNX models -- as a Python class and a
FastAPI server.

## Layout

| File | Purpose |
| --- | --- |
| [realesrgan_onnx.py](realesrgan_onnx.py) | `RealESRGANOnnx` -- the upsampler (preprocess -> session -> postprocess) |
| [server.py](server.py) | FastAPI server |
| [test.py](test.py) | Smoke-tests the API against an image you pass in |
| `models/` | `.onnx` weights (not tracked in git) |
| `test/` | Sample images used by the smoke checks below (not tracked in git) |

## Install

```bash
pip install -r requirements.txt
```

Python 3.12. `requirements.txt` pins `onnxruntime-gpu`, which needs an NVIDIA driver
with CUDA 12 + cuDNN 9. For a CPU-only machine, install `onnxruntime` instead and
start the server with `--device cpu`.

## Library

```python
import cv2
from realesrgan_onnx import RealESRGANOnnx

upsampler = RealESRGANOnnx('models/RealESRGAN_x2plus.onnx', device='cuda')
output = upsampler.infer(cv2.imread('input.jpg'))            # BGR uint8 in, BGR uint8 out
outputs = upsampler.infer_batch([img1, img2])                # needs a model exported with --dynamic
```

- fp32 and fp16 models are both supported; the input dtype is read from the model.
- `device` is `'cuda'` (default) or `'cpu'`; `'cuda'` falls back to CPU when no CUDA
  execution provider is available.
- `infer_batch` requires all images to share one shape.

## Server

```bash
python server.py
python server.py -m RealESRGAN_x4plus_fp16.onnx -d cpu -p 9000
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `-m`, `--model` | `RealESRGAN_x2plus.onnx` | Upscaling model file name |
| `-d`, `--device` | `cuda` | `cuda` or `cpu` |
| `--max_side` | `1920` | Images with a longer side than this are downscaled before inference |
| `--host` | `0.0.0.0` | Bind address |
| `-p`, `--port` | `8080` | Bind port |

The model directory is always `models/`, so `--model` take a file name, not a path.
The model is loaded once at startup and a missing file fails immediately, so startup
takes a few seconds and the port only opens once the session is ready.

An upload whose longer side exceeds `--max_side` is downscaled to that limit (aspect
ratio kept), upsampled, then resized back to its **original** dimensions -- so oversized images
come back the same size they went in, enhanced rather than enlarged. Images within
the limit are upsampled normally and come back at the model's scale.

## API

Interactive docs at `/docs`.

### `GET /api/health/`

```bash
curl http://127.0.0.1:8080/api/health/
```

```json
{"status": "ok", "model": "RealESRGAN_x2plus.onnx", "device": "cuda", "max_side": 1920,
 "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
```

`providers` comes from the live session, so it shows whether CUDA actually engaged or
fell back to CPU. `503` if the model is not loaded.

### `POST /api/upscale/`

Request -- `multipart/form-data`:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `image` | file | required | Image to upscale |

Response -- the upscaled image as raw `image/png` bytes at the input's original
dimensions. `400` if the upload cannot be decoded as an image, `500` if the result
cannot be encoded.

```bash
curl -X POST -F "image=@test/cat.jpg" http://127.0.0.1:8080/api/upscale/ -o test/cat_out.png
```

## Docker

Base image `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04`, so run it with `--gpus all`.
`models/` is not baked into the image (~500 MB of onnx) -- mount it at run time.

```bash
docker build -t realesrgan-server .
docker run --gpus all -p 8080:8080 -v ./models:/app/models realesrgan-server
```

Server flags pass straight through the entrypoint:

```bash
docker run --gpus all -p 8080:8080 -v ./models:/app/models realesrgan-server -m RealESRGAN_x4plus_fp16.onnx --max_side 1280
```

Omit `--gpus all` and pass `-d cpu` to run on CPU. On Windows use an absolute path for
the mount, e.g. `-v "E:\Project\phoenix\servers\Real-ESRGAN\models:/app/models"`.

To bake the models into the image instead, drop `models/` from `.dockerignore` and add
`COPY models/*.onnx ./models/` to the Dockerfile.

### GPU notes

`--gpus all` requires Docker Desktop's **WSL2 backend** (Settings -> General -> *Use the
WSL 2 based engine*). On the Hyper-V backend the container has no NVIDIA driver and
fails with `nvidia-container-cli: initialization error: load library failed:
libnvidia-ml.so.1`.

## Test

`test.py` takes the input image path as its argument and checks the health route, a
round-trip of that image, and that a non-image upload is rejected. It exits non-zero
if any check fails. Each line carries the response time for that request, so you can
see what the model actually costs per call. It needs `requests`, which is deliberately
kept out of `requirements.txt` so it stays out of the Docker image.

```bash
pip install requests
python server.py --device cpu &
python test.py test/cat.jpg
```

```
[PASS] health -- 5ms -- {'status': 'ok', 'model': 'RealESRGAN_x2plus.onnx', ...}
[PASS] upscale -- 33089ms -- (438, 500, 3) -> (876, 1000, 3)
       wrote test/cat_out.png
[PASS] rejects a non-image -- 25ms -- 400 {"detail":"could not decode image"}
3/3 checks passed
```

Those times are from a CPU run on this sample; `--device cuda` and the fp16 graphs
give very different numbers.

| Flag | Default | Meaning |
| --- | --- | --- |
| `image` | required | Path to the image to send |
| `--url` | `http://127.0.0.1:8080` | Base url of the running server |
| `-o`, `--output` | `test/cat_out.png` | Where to write the returned png; defaults to the input path with `_out.png` in place of its extension |

`cat.jpg` is under `--max_side`, so it comes back at the model's scale rather
than the input size -- that is the expected result, not a failure.

Point it at another host or a container with `--url`:

```bash
python test.py test/cat.jpg --url http://127.0.0.1:9000
```

### With curl

The same three checks by hand, without `test.py` or `requests`:

```bash
curl -s http://127.0.0.1:8080/api/health/
# {"status": "ok", ...}

curl -s -X POST -F "image=@test/cat.jpg" http://127.0.0.1:8080/api/upscale/ -o test/cat_out.png
python -c "import cv2; print(cv2.imread('test/cat.jpg').shape, '->', cv2.imread('test/cat_out.png').shape)"
# (438, 500, 3) -> (876, 1000, 3)

curl -s -o /dev/null -w '%{http_code}\n' -X POST -F "image=@README.md" http://127.0.0.1:8080/api/upscale/
# 400
```

## Notes

- One ONNX session is created at startup and shared. FastAPI runs the endpoint in a
  threadpool, so concurrent requests are correct but throughput is bounded by the
  single session.
- The `_fp16` graphs are selected by name, e.g. `--model RealESRGAN_x4plus_fp16.onnx`.
- The `.pth` files in `models/` are the original torch checkpoints the ONNX graphs were
  exported from. They are unused at inference time and excluded from the Docker image.

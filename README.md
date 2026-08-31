# Real-ESRGAN

Minimal Real-ESRGAN image upscaling: PyTorch inference, ONNX export, and ONNX Runtime inference.

The upsampler classes live in their own modules, with a thin CLI on top of each:

| | class | CLI |
| --- | --- | --- |
| PyTorch | [realesrgan.py](realesrgan.py) — `RealESRGAN` | [inference.py](inference.py) |
| ONNX Runtime | [realesrgan_onnx.py](realesrgan_onnx.py) — `RealESRGANOnnx` | [inference_onnx.py](inference_onnx.py) |

[export_onnx.py](export_onnx.py) exports a `.pth` checkpoint to `.onnx`.

## Install

```bash
pip install -r requirements.txt
```

For GPU ONNX inference, install `onnxruntime-gpu` instead of `onnxruntime`.

## Weights

Put the pretrained `.pth` files in `weights/`. Download them from the
[Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases):

| Model | Scale | `--num_block` |
| --- | --- | --- |
| `RealESRGAN_x2plus.pth` | 2 | 23 |
| `RealESRGAN_x4plus.pth` | 4 | 23 |
| `RealESRNet_x4plus.pth` | 4 | 23 |
| `RealESRGAN_x4plus_anime_6B.pth` | 4 | 6 |

## Usage

### PyTorch

```bash
python inference.py -i test/00003.png -m weights/RealESRGAN_x4plus.pth -s 4
```

The anime model has 6 RRDB blocks instead of 23:

```bash
python inference.py -i test/00003.png -m weights/RealESRGAN_x4plus_anime_6B.pth -s 4 --num_block 6
```

Options: `-i/--input` (required), `-o/--output` (folder, default `results`),
`-m/--model_path`, `-s/--scale`, `--num_block`.

Results are written to `<output>/<name>_x<scale><ext>`, e.g. `results/00003_x4.png`.
CUDA is used when available, otherwise CPU.

### Export to ONNX

```bash
python export_onnx.py -m weights/RealESRGAN_x4plus.pth -s 4
```

Writes `weights/RealESRGAN_x4plus.onnx` (override with `-o`). Height and width are
always dynamic, so one exported model handles any input resolution. Pass `--dynamic`
to make the batch dimension dynamic too, which is required for batched ONNX inference.

### ONNX Runtime

```bash
python inference_onnx.py -i test/00003.png -m weights/RealESRGAN_x4plus.onnx
```

The scale is inferred from the output size, so there is no `-s` flag. Uses
`CUDAExecutionProvider` when available, falling back to `CPUExecutionProvider`.

## fp32 vs fp16

`RealESRGAN_x2plus`, a 512x353 input, `CUDAExecutionProvider` on an RTX 4070,
5 warmup runs then 20 timed runs of `infer()`:

| | mean | std | min | max |
| --- | --- | --- | --- | --- |
| fp32 | 183.6 ms | 0.9 | 181.5 | 185.0 |
| fp16 | 117.0 ms | 1.1 | 115.5 | 120.0 |

fp16 is **1.57x faster**, at a max per-pixel difference of 1/255 (mean 0.038) —
under half a quantization step, so the output is visually identical.

Timings are end-to-end `infer()`, including the float32 pre/postprocess that both
paths share, so the session-only speedup is higher. The ratio also shifts with
resolution and GPU.

## As a library

Both classes expose `infer` (one image) and `infer_batch` (a list of images that
share one shape). Images are BGR `uint8` HWC arrays, the
format `cv2.imread` returns.

```python
import cv2
from realesrgan import RealESRGAN

upsampler = RealESRGAN('weights/RealESRGAN_x4plus.pth', scale=4)
out = upsampler.infer(cv2.imread('test/00003.png', cv2.IMREAD_COLOR))
cv2.imwrite('results/00003_x4.png', out)
```

```python
from realesrgan_onnx import RealESRGANOnnx

upsampler = RealESRGANOnnx('weights/RealESRGAN_x4plus.onnx')
out = upsampler.infer(cv2.imread('test/00003.png', cv2.IMREAD_COLOR))
```

## Notes

- The whole image is processed in one pass — there is no tiling, so large inputs
  are limited by available GPU memory.
- Batched ONNX inference requires a model exported with `--dynamic`.
- The x2 models `pixel_unshuffle` by 2, so they need even input dimensions. Odd
  inputs are padded by one edge pixel and the output is cropped back, so the
  result is always exactly `scale` times the input size.
- ONNX models may be fp32 or fp16; the input dtype is read from the model.

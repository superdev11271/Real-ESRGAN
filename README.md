# Real-ESRGAN

Minimal Real-ESRGAN image upscaling: PyTorch inference, ONNX export, and ONNX Runtime inference.

- [inference.py](inference.py) — upscale an image with a `.pth` checkpoint (PyTorch)
- [export_onnx.py](export_onnx.py) — export a checkpoint to `.onnx`
- [inference_onnx.py](inference_onnx.py) — upscale an image with an exported `.onnx` model

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

## As a library

Both entry points expose a class with `infer` (one image) and `infer_batch`
(a list of images that share one shape). Images are BGR `uint8` HWC arrays, the
format `cv2.imread` returns.

```python
import cv2
from inference import RealESRGAN

upsampler = RealESRGAN('weights/RealESRGAN_x4plus.pth', scale=4)
out = upsampler.infer(cv2.imread('test/00003.png', cv2.IMREAD_COLOR))
cv2.imwrite('results/00003_x4.png', out)
```

```python
from inference_onnx import RealESRGANOnnx

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

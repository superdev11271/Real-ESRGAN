"""FastAPI server exposing Real-ESRGAN ONNX upscaling at POST /api/upscale/.

Start with the four models and a device; the models always live in `MODEL_DIR`:

    python server.py -m1 RealESRGAN_x2plus_fp16.onnx -m2 RealESRGAN_x4plus_fp16.onnx --device cuda

All four are loaded at startup and each request picks one by name with the `model`
form field, defaulting to the first one loaded. A slot whose file is missing is
skipped, and the server exits if that leaves nothing to serve. POST an image
(multipart field `image`) and get the upscaled image back as PNG. Images whose
longest side exceeds `--max_side` are downscaled for inference and scaled back up
to their original size. GET /api/health/ for a liveness check and the available
model names.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from realesrgan_onnx import RealESRGANOnnx

MODEL_DIR = 'models'
# the first slot that actually loads is the default requests fall back to
DEFAULT_MODELS = ['RealESRGAN_x2plus_fp16.onnx', 'RealESRGAN_x4plus_anime_6B_fp16.onnx',
                  'RealESRGAN_x4plus_fp16.onnx', 'RealESRNet_x4plus_fp16.onnx']
DEFAULT_DEVICE = 'cuda'
DEFAULT_MAX_SIDE = 1280

app = FastAPI(title='Real-ESRGAN ONNX')
upsamplers = {}
model_info = {}
max_side = DEFAULT_MAX_SIDE


def default_model():
    """Name of the first model that loaded, which requests fall back to."""
    return next(iter(upsamplers))


@app.get('/api/health/')
async def health():
    """Report which models are loaded and how they are configured."""
    if not upsamplers:
        raise HTTPException(status_code=503, detail='models not loaded')
    return {'status': 'ok', 'models': list(upsamplers), 'default_model': default_model(),
            **model_info, 'providers': upsamplers[default_model()].session.get_providers()}


@app.post('/api/upscale/')
async def upscale(image: UploadFile = File(...), model: str = Form(None)):
    """Upscale the uploaded image with the named model and return it as PNG."""
    name = model or default_model()
    upsampler = upsamplers.get(name)
    if upsampler is None:
        raise HTTPException(status_code=400,
                            detail=f'unknown model {name!r}, available: {list(upsamplers)}')

    img = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail='could not decode image')

    h, w = img.shape[:2]
    if max(h, w) > max_side:
        # shrink the long side to max_side, upsample, then come back to the original size
        ratio = max_side / max(h, w)
        small = cv2.resize(img, (round(w * ratio), round(h * ratio)), interpolation=cv2.INTER_AREA)
        output = cv2.resize(upsampler.infer(small), (w, h), interpolation=cv2.INTER_AREA)
    else:
        output = upsampler.infer(img)
    ok, buf = cv2.imencode('.png', output)
    if not ok:
        raise HTTPException(status_code=500, detail='could not encode result')
    return Response(content=buf.tobytes(), media_type='image/png')


def main():
    parser = argparse.ArgumentParser()
    for slot, default in enumerate(DEFAULT_MODELS, start=1):
        parser.add_argument(f'-m{slot}', f'--model{slot}', type=str, default=default,
                            help=f'model {slot} file name inside {MODEL_DIR}/')
    parser.add_argument('-d', '--device', type=str, default=DEFAULT_DEVICE, choices=['cuda', 'cpu'])
    parser.add_argument('--max_side', type=int, default=DEFAULT_MAX_SIDE,
                        help='images with a longer side than this are downscaled before inference')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('-p', '--port', type=int, default=8080)
    args = parser.parse_args()

    global max_side
    for slot in range(1, len(DEFAULT_MODELS) + 1):
        file_name = getattr(args, f'model{slot}')
        model_path = os.path.join(MODEL_DIR, file_name)
        # a slot whose file is missing is skipped, so the server still comes up on
        # whichever models are actually present -- named or defaulted alike
        if not os.path.isfile(model_path):
            print(f'-m{slot}: {model_path} not found, skipping', file=sys.stderr)
            continue
        # requests name a model by its file name without the .onnx suffix
        upsamplers[os.path.splitext(file_name)[0]] = RealESRGANOnnx(model_path, device=args.device)
    if not upsamplers:
        parser.error(f'no models found in {MODEL_DIR}/, nothing to serve')
    max_side = args.max_side
    model_info.update(device=args.device, max_side=args.max_side)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

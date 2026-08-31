"""FastAPI server exposing Real-ESRGAN ONNX upscaling at POST /api/upscale/."""
import argparse
import os

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from realesrgan_onnx import RealESRGANOnnx

MODEL_DIR = 'models'

app = FastAPI(title='Real-ESRGAN ONNX')
upsampler = None
model_name = None
max_side = 1920


@app.get('/api/health/')
async def health():
    """Report whether the model is loaded and which providers it runs on."""
    if upsampler is None:
        raise HTTPException(status_code=503, detail='model not loaded')
    return {
        'status': 'ok',
        'model': model_name,
        'providers': upsampler.session.get_providers(),
    }


@app.post('/api/upscale/')
async def upscale(image: UploadFile = File(...)):
    """Upscale the uploaded image and return it as PNG."""
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
    parser.add_argument('-m', '--model', type=str, default='RealESRGAN_x2plus.onnx',
                        help=f'model file name inside {MODEL_DIR}/')
    parser.add_argument('-d', '--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--max_side', type=int, default=1920,
                        help='images with a longer side than this are downscaled before inference')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('-p', '--port', type=int, default=8080)
    args = parser.parse_args()

    global upsampler, model_name, max_side
    model_path = os.path.join(MODEL_DIR, args.model)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    upsampler = RealESRGANOnnx(model_path, device=args.device)
    model_name = args.model
    max_side = args.max_side

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

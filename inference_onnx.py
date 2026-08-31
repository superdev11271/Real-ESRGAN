"""CLI for Real-ESRGAN ONNX inference: read -> upsample -> write."""
import argparse
import os

import cv2

from realesrgan_onnx import RealESRGANOnnx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='input image path')
    parser.add_argument('-o', '--output', type=str, default='results', help='output folder')
    parser.add_argument('-m', '--model_path', type=str, default='models/RealESRGAN_x4plus.onnx')
    args = parser.parse_args()

    upsampler = RealESRGANOnnx(args.model_path)
    os.makedirs(args.output, exist_ok=True)

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.input)
    output = upsampler.infer(img)

    # save as <original name>_x<scale>.<original ext>, e.g. 0014_x4.jpg
    scale = output.shape[0] // img.shape[0]
    basename, ext = os.path.splitext(os.path.basename(args.input))
    save_path = os.path.join(args.output, f'{basename}_x{scale}{ext}')
    cv2.imwrite(save_path, output)
    print(f'{args.input} -> {save_path} {output.shape[1]}x{output.shape[0]}')


if __name__ == '__main__':
    main()

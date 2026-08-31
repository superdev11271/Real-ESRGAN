"""CLI for Real-ESRGAN PyTorch inference: read -> upsample -> write."""
import argparse
import os

import cv2

from realesrgan import RealESRGAN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='input image path')
    parser.add_argument('-o', '--output', type=str, default='results', help='output folder')
    parser.add_argument('-m', '--model_path', type=str, default='weights/RealESRGAN_x2plus.pth')
    parser.add_argument('-s', '--scale', type=int, default=2, help='native scale of the network')
    parser.add_argument('--num_block', type=int, default=23, help='6 for the anime_6B model')
    args = parser.parse_args()

    upsampler = RealESRGAN(args.model_path, scale=args.scale, num_block=args.num_block)
    os.makedirs(args.output, exist_ok=True)

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.input)
    output = upsampler.infer(img)

    # save as <original name>_x<scale>.<original ext>, e.g. 0014_x4.jpg
    basename, ext = os.path.splitext(os.path.basename(args.input))
    save_path = os.path.join(args.output, f'{basename}_x{args.scale}{ext}')
    cv2.imwrite(save_path, output)
    print(f'{args.input} -> {save_path} {output.shape[1]}x{output.shape[0]}')


if __name__ == '__main__':
    main()

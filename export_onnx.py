"""Export a Real-ESRGAN RRDBNet checkpoint to ONNX."""
import argparse
import os

import torch
from basicsr.archs.rrdbnet_arch import RRDBNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, default='weights/RealESRGAN_x4plus.pth')
    parser.add_argument('-o', '--output', type=str, default=None, help='output .onnx path')
    parser.add_argument('-s', '--scale', type=int, default=4, help='native scale of the network')
    parser.add_argument('--num_block', type=int, default=23, help='6 for the anime_6B model')
    parser.add_argument('--dynamic', action='store_true', help='also make the batch dimension dynamic')
    args = parser.parse_args()

    # default: alongside the checkpoint, e.g. weights/RealESRGAN_x4plus.onnx
    output = args.output or os.path.splitext(args.model_path)[0] + '.onnx'

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=args.num_block, num_grow_ch=32, scale=args.scale)
    loadnet = torch.load(args.model_path, map_location='cpu')
    keyname = 'params_ema' if 'params_ema' in loadnet else 'params'
    model.load_state_dict(loadnet[keyname], strict=True)
    model.eval()

    # NCHW RGB float in [0, 1], same as the pytorch preprocess
    x = torch.rand(1, 3, 64, 64)

    # H/W are always dynamic so one model serves any input resolution
    axes = {2: 'height', 3: 'width'}
    if args.dynamic:
        axes[0] = 'batch'
    dynamic_axes = {'input': dict(axes), 'output': dict(axes)}

    with torch.no_grad():
        torch.onnx.export(
            model,
            x,
            output,
            opset_version=11,
            export_params=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes)
    print(f'{args.model_path} -> {output}')


if __name__ == '__main__':
    main()

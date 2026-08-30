"""Minimal Real-ESRGAN inference: read -> preprocess -> model -> postprocess -> write."""
import argparse
import os

import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet

class RealESRGAN():
    """Upsample images with a Real-ESRGAN RRDBNet model.

    Args:
        model_path (str): Path to the pretrained .pth weights.
        scale (int): Native upscale factor of the network. Default: 4.
        num_block (int): Number of RRDB blocks. 23 for the x4plus/x2plus models,
            6 for RealESRGAN_x4plus_anime_6B. Default: 23.
        device (torch.device): Defaults to cuda if available, else cpu.
    """

    def __init__(self, model_path, scale=4, num_block=23, device=None):
        self.scale = scale
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=num_block, num_grow_ch=32, scale=scale)
        loadnet = torch.load(model_path, map_location='cpu')
        keyname = 'params_ema' if 'params_ema' in loadnet else 'params'
        model.load_state_dict(loadnet[keyname], strict=True)
        model.eval()
        self.model = model.to(self.device)

    def preprocess(self, imgs):
        """List of BGR uint8 HWC [0, 255] -> RGB float NCHW [0, 1] on device."""
        batch = []
        for img in imgs:
            img = img.astype(np.float32) / 255.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch.append(np.transpose(img, (2, 0, 1)))
        return torch.from_numpy(np.stack(batch)).to(self.device)

    def postprocess(self, output):
        """RGB float NCHW [0, 1] -> list of BGR uint8 HWC [0, 255]."""
        output = output.clamp_(0, 1).cpu().numpy()
        imgs = []
        for img in output:
            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            imgs.append((img * 255.0).round().astype(np.uint8))
        return imgs

    def infer(self, img):
        """Upsample one BGR uint8 image and return a BGR uint8 image."""
        return self.infer_batch([img])[0]

    @torch.no_grad()
    def infer_batch(self, imgs):
        """Upsample a list of BGR uint8 images sharing the same shape."""
        shapes = {img.shape for img in imgs}
        assert len(shapes) == 1, f'batched images must share one shape, got {shapes}'
        return self.postprocess(self.model(self.preprocess(imgs)))


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

"""Minimal Real-ESRGAN ONNX inference: read -> preprocess -> session -> postprocess -> write."""
import argparse
import os

import cv2
import numpy as np
import onnxruntime

class RealESRGANOnnx():
    """Upsample images with an exported Real-ESRGAN ONNX model.

    Args:
        model_path (str): Path to the .onnx model. fp32 and fp16 models are
            both supported; the input dtype is read from the model.
        providers (list): onnxruntime execution providers. Defaults to CUDA if
            available, else CPU.
    """

    def __init__(self, model_path, providers=None):
        if providers is None:
            available = onnxruntime.get_available_providers()
            providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in available else []
            providers.append('CPUExecutionProvider')
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # fp16 models expect a float16 tensor; everything else stays float32
        self.dtype = np.float16 if inp.type == 'tensor(float16)' else np.float32

    def preprocess(self, imgs):
        """List of BGR uint8 HWC [0, 255] -> RGB NCHW [0, 1] in the model's dtype."""
        batch = []
        for img in imgs:
            img = img.astype(np.float32) / 255.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch.append(np.transpose(img, (2, 0, 1)))
        return np.stack(batch).astype(self.dtype)

    def postprocess(self, output):
        """RGB float NCHW [0, 1] -> list of BGR uint8 HWC [0, 255]."""
        output = np.clip(output.astype(np.float32), 0, 1)
        imgs = []
        for img in output:
            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            imgs.append((img * 255.0).round().astype(np.uint8))
        return imgs

    def infer(self, img):
        """Upsample one BGR uint8 image and return a BGR uint8 image."""
        return self.infer_batch([img])[0]

    def infer_batch(self, imgs):
        """Upsample a list of BGR uint8 images sharing the same shape.

        A batch larger than 1 needs a model exported with `--dynamic`.
        """
        shapes = {img.shape for img in imgs}
        assert len(shapes) == 1, f'batched images must share one shape, got {shapes}'
        x = self.preprocess(imgs)
        # x2 models pixel_unshuffle by 2, so H/W must be even: pad, then crop back
        h, w = x.shape[2:]
        x = np.pad(x, ((0, 0), (0, 0), (0, h % 2), (0, w % 2)), mode='edge')
        output = self.session.run(None, {self.input_name: x})[0]
        scale = output.shape[2] // x.shape[2]
        return self.postprocess(output[:, :, :scale * h, :scale * w])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='input image path')
    parser.add_argument('-o', '--output', type=str, default='results', help='output folder')
    parser.add_argument('-m', '--model_path', type=str, default='weights/RealESRGAN_x4plus.onnx')
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

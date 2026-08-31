"""Real-ESRGAN ONNX upsampler: preprocess -> session -> postprocess."""
import cv2
import numpy as np
import onnxruntime

class RealESRGANOnnx():
    """Upsample images with an exported Real-ESRGAN ONNX model.

    Args:
        model_path (str): Path to the .onnx model. fp32 and fp16 models are
            both supported; the input dtype is read from the model.
        device (str): 'cuda' (default) or 'cpu'. 'cuda' falls back to CPU when
            no CUDA execution provider is available.
    """

    def __init__(self, model_path, device='cuda'):
        if device not in ('cuda', 'cpu'):
            raise ValueError(f"device must be 'cuda' or 'cpu', got {device!r}")
        providers = ['CPUExecutionProvider']
        if device == 'cuda' and 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
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

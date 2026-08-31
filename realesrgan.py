"""Real-ESRGAN upsampler: preprocess -> model -> postprocess."""
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
        x = self.preprocess(imgs)
        # x2 models pixel_unshuffle by 2, so H/W must be even: pad, then crop back
        h, w = x.shape[2:]
        x = torch.nn.functional.pad(x, (0, w % 2, 0, h % 2), mode='replicate')
        output = self.model(x)
        return self.postprocess(output[:, :, :self.scale * h, :self.scale * w])

import cog
import torch

from configs.config_IFAN import get_config
from ckpt_manager import CKPT_Manager
from models import create_model

from utils import *
from data_loader.utils import load_file_list, refine_image, read_frame

from pathlib import Path
import tempfile
import cv2

class Predictor(cog.Predictor):
    def setup(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.config = get_config('IFAN_CVPR2021', 'IFAN', 'config_IFAN')
        self.config.network = 'IFAN'
        model = create_model(self.config)
        self.network = model.get_network().eval()
        self.network = self.network.to(self.device)

        ckpt_manager = CKPT_Manager(root_dir='', model_name='IFAN')
        load_state, ckpt_name = ckpt_manager.load_ckpt(self.network, abs_name = './ckpt/IFAN.pytorch')

    @cog.input("image", type=Path, help="Input image, only supports images with .png and .jpg extensions")
    def predict(self, image):
        max_side = 1920
        assert str(image).split('.')[-1] in ['png', 'jpg'], 'image should end with ".jpg" or ".png"'

        C_cpu = read_frame(str(image), self.config.norm_val, None)

        b, h, w, c = C_cpu.shape
        if max(h, w) > max_side:
            scale_ratio = max_side / max(h, w)
            C_cpu = np.expand_dims(cv2.resize(C_cpu[0], dsize=(int(w*scale_ratio), int(h*scale_ratio)), interpolation=cv2.INTER_AREA), 0)

        C = torch.FloatTensor(refine_image(C_cpu, self.config.refine_val).transpose(0, 3, 1, 2).copy()).to(self.device)

        with torch.no_grad():
            out = self.network(C)

        output = out['result']
        output_cpu = output.cpu().numpy()[0].transpose(1, 2, 0)
        output_cpu = (np.flip(output_cpu, 2) * 255).astype(np.uint8)

        out_path = Path(tempfile.mkdtemp()) / 'out.jpg'
        cv2.imwrite(str(out_path), output_cpu)

        return out_path

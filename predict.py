from cog import BasePredictor, Path, Input
from typing import List
import torch
import utils.imgops as ops
import utils.architecture.architecture as arch
import cv2
import numpy as np
import torch
import shutil
import sys

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        NORMAL_MAP_MODEL = 'utils/models/1x_NormalMapGenerator-CX-Lite_200000_G.pth'
        OTHER_MAP_MODEL = 'utils/models/1x_FrankenMapGenerator-CX-Lite_215000_G.pth'

        def load_model(model_path):
            self.device = torch.device('cuda')
            state_dict = torch.load(model_path)
            model = arch.RRDB_Net(3, 3, 32, 12, gc=32, upscale=1, norm_type=None, act_type='leakyrelu',
                                    mode='CNA', res_scale=1, upsample_mode='upconv')
            model.load_state_dict(state_dict, strict=True)
            del state_dict
            model.eval()
            for k, v in model.named_parameters():
                v.requires_grad = False
            return model.to(self.device)
        self.models = [
            # NORMAL MAP
            load_model(NORMAL_MAP_MODEL), 
            # ROUGHNESS/DISPLACEMENT MAPS
            load_model(OTHER_MAP_MODEL)
            ]

    def predict(self,
            image: Path = Input(description="RGB Image to generate maps from"),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        shutil.copyfile(image, "/tmp/image.png")
        img_path = "/tmp/image.png"

        try: 
            img = cv2.imread(img_path)
        except:
            img = cv2.imread(img_path)

            
        # Seamless modes
        img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_WRAP)
        # Make maps
        def process(self,img, model):
            img = img * 1. / np.iinfo(img.dtype).max
            img = img[:, :, [2, 1, 0]]
            img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
            img_LR = img.unsqueeze(0)
            img_LR = img_LR.to(self.device)

            output = model(img_LR).data.squeeze(
                0).float().cpu().clamp_(0, 1).numpy()
            output = output[[2, 1, 0], :, :]
            output = np.transpose(output, (1, 2, 0))
            output = (output * 255.).round()
            return output
        
        rlts = [process(self,img, model) for model in self.models]

        # ... pre-processing ...
        normal_map = rlts[0]
        roughness = rlts[1][:, :, 1]
        displacement = rlts[1][:, :, 0]

        output_paths = []
        for index, item in enumerate([normal_map,roughness,displacement]):
            output_path = f"/tmp/out-{index}.png"
            cv2.imwrite(output_path, item)
            output_paths.append(Path(output_path))
            
        return output_paths
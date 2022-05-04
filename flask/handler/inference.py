import torch
import numpy as np
import cv2

from handler.arch import net_G

class ESRGANHandler:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = net_G.RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4).to(self.device)
        self.state_dict = torch.load(model_path)
        self.model.load_state_dict(self.state_dict['params_ema'])
        self.model.eval()
    
    def preprocessing(self, img):
        alpha = None

        if np.max(img) > 256:
            max_range = 65535
        else:
            max_range = 255
        
        img = img / max_range

        if len(img.shape) == 2: # gray image
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        elif img.shape[0] == 4: # RGBA
            img_mode = 'RGBA'
            alpha = img[3, :, :]
            img = img[0:3, :, :]
          
        else:
            img_mode = 'RGB'

        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0).to(self.device)
        
        return img, img_mode, max_range, alpha
    
    def inference(self, img):
        with torch.no_grad():
            output = self.model(img)
        return output
    
    def post_process(self, output, img_mode, max_range, alpha=None):
        output = output.detach().squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output, (1, 2, 0))

        if img_mode == 'L':
            output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        
        elif alpha and img_mode == 'RGBA':
            alpha, _, _ = self.preprocessing(alpha)
            alpha = self.inference(alpha)
            output_alpha = alpha.detach().squeeze().float().cpu().clamp_(0, 1).numpy()
            output_alpha = np.transpose(output_alpha, (1, 2, 0))
            output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)

            output = cv2.cvtColor(output, cv2.COLOR_BGR2BGRA)
            output[:, :, 3] = output_alpha
        
        if max_range == 65535:
            output = (output * 65535.0).round().astype(np.uint8)
        else:
            output = (output * 255.0).round().astype(np.uint8)
        
        return output

    def handle(self, img):
        img, img_mode, max_range, alpha = self.preprocessing(img)
        output = self.inference(img)
        output = self.post_process(output, img_mode, max_range, alpha)

        return output

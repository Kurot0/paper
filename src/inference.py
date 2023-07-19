import os
import torch
import yaml
import argparse
import importlib
import numpy as np
from PIL import Image
from ssimloss import SSIMLoss


class Inferencer:
    def __init__(self, params):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.params = params

        data_path = self.params['data_path']
        self.x_test = torch.load(data_path + '/x_test.pt').float().to(self.device)
        self.y_test = torch.load(data_path + '/y_test.pt').float().to(self.device)
        self.y_pred_path = data_path + '/y_pred.pt'

        model_module = importlib.import_module(params['model_module'])
        model_class = getattr(model_module, params['model_class'])

        self.model = model_class(**self.params).to(self.device)

        if os.path.isfile(self.params['checkpoint_path'] + '/best.pth'):
            self.model.load_state_dict(torch.load(self.params['checkpoint_path'] + '/best.pth'))
        self.model.eval()

        self.psnr = None
        self.ssim = None

        self.mask = torch.from_numpy(np.array(Image.open(params['mask_path']))).float().to(self.device)

    def inference(self):
        with torch.no_grad():
            y_pred = self.model(self.x_test)
            torch.save(y_pred.cpu(), self.y_pred_path)

            y_test_masked = self.y_test * self.mask
            y_pred_masked = y_pred * self.mask

            mean_squared_error = torch.nn.MSELoss()
            mse_loss = mean_squared_error(y_test_masked, y_pred_masked)

            max_pixel_value = torch.max(y_test_masked).item()
            min_pixel_value = torch.min(y_test_masked).item()
            pixel_range = max_pixel_value - min_pixel_value

            self.psnr = 20 * torch.log10(pixel_range / torch.sqrt(mse_loss))
            self.psnr = self.psnr.cpu().detach().numpy()

            ssim_loss = SSIMLoss()
            self.ssim = -ssim_loss(y_test_masked, y_pred_masked)
            self.ssim = self.ssim.cpu().detach().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='paper/config.yaml', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        params = yaml.safe_load(stream)

    inferencer = Inferencer(params)
    inferencer.inference()

    print(f'PSNR: {inferencer.psnr}')
    print(f'SSIM: {inferencer.ssim}')

if __name__ == '__main__':
    main()

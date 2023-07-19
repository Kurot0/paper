import os
import yaml
import argparse
import torch
import numpy as np
from train import Trainer
from inference import Inferencer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='paper/config.yaml', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        params = yaml.safe_load(stream)
    
    params['early_stop_flag'] = 1
    params['cross_validation_flag'] = 1

    psnr_results = []
    ssim_results = []

    for i in range(10):
        dir_name = 'crossValidation_all{}'.format(i)
        print(f"Starting cross validation {i}")
        
        params['data_path'] = os.path.join('paper/data/cv_data', dir_name)
        
        trainer = Trainer(params)
        trainer.train()

        inferencer = Inferencer(params)
        if trainer.early_stop_model is not None:
            inferencer.model.load_state_dict(trainer.early_stop_model)
        else:
            inferencer.model.load_state_dict(trainer.best_model)
        inferencer.model.eval()
        inferencer.inference()
        
        psnr_results.append(inferencer.psnr)
        ssim_results.append(inferencer.ssim)

        del trainer
        del inferencer
        torch.cuda.empty_cache()
    
    print(f"Average PSNR over cross validations: {np.mean(psnr_results)}")
    print(f"Average SSIM over cross validations: {np.mean(ssim_results)}")

if __name__ == '__main__':
    main()

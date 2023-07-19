import os
import numpy as np
import matplotlib.pyplot as plt
import torch

output_dir = 'paper/data/images'
os.makedirs(output_dir, exist_ok=True)

data_4d = torch.load('paper/data/exp_data/y_pred.pt').numpy()

for i in range(data_4d.shape[0]):
    data = data_4d[i, 0]

    plt.imshow(data, cmap='Reds_r', interpolation='nearest')
    plt.colorbar()

    filename = f'{i+1:03}.png'  

    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

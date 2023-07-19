import pandas as pd
import glob
import numpy as np
import torch


path = 'paper/data/raw_data/2013_Sv05s_LL/*.dat'
files = sorted(glob.glob(path))

df = pd.read_csv(files[0], names=['lon', 'lat', 'amp', 'sid'])
lons = np.unique(df['lon'].values)
lats = np.unique(df['lat'].values)[::-1]

data_4d = np.zeros((len(files), 1, len(lats), len(lons)))

for file_index, file in enumerate(files):
    df = pd.read_csv(file, names=['lon', 'lat', 'amp', 'sid'])

    for ind in range(len(df)):

        lonInd = int(np.where(df['lon'][ind] == lons)[0])
        latInd = int(np.where(df['lat'][ind] == lats)[0]) 

        data_4d[file_index, 0, latInd, lonInd] = df['amp'][ind]

    print(file_index, file)

data_4d = np.pad(data_4d, ((0, 0), (0, 0), (0, 0), (2, 2)), mode='constant')

data_4d = np.delete(data_4d, np.s_[:3], axis=2)
data_4d = np.delete(data_4d, np.s_[-3:], axis=2)

data_min = np.min(data_4d, axis=(0, 2, 3), keepdims=True)
data_max = np.max(data_4d, axis=(0, 2, 3), keepdims=True)
data_4d = (data_4d - data_min) / (data_max - data_min)

data_4d = torch.from_numpy(data_4d)

torch.save(data_4d, 'paper/data/prep_data/2013_Sv05s_LL.pt')

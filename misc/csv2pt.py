import csv
import glob
import numpy as np
import torch


path = "paper/data/raw_data/2013_calc_scenario/*.csv"
files = sorted(glob.glob(path))

max_data = np.genfromtxt("paper/data/raw_data/2013_calc_scenario/TF111_h01-add-ase_nm-vr2_fm1_sd1.csv", delimiter=",")
lat = np.unique(max_data[:,0])[::-1]
lon = np.unique(max_data[:,1])

data_4d = np.zeros((len(files), max_data.shape[1]-3, len(lat), len(lon)))

for file_index, file in enumerate(files):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = [row + [0.]*(max_data.shape[1]-len(row)) for row in reader]
    data = np.array(data, dtype=float)

    data[:,6] = data[:,6] * 10**data[:,7]
    data = np.delete(data, 7, 1)

    for row in range(data.shape[0]):
        lat_index = np.where(lat == data[row,0])[0][0]
        lon_index = np.where(lon == data[row,1])[0][0]
        data_4d[file_index, :data.shape[1]-2, lat_index, lon_index] = data[row, 2:]

    print(file_index, file)

data_4d = np.pad(data_4d, ((0, 0), (0, 0), (0, 12), (0, 13)), mode='constant')

data_min = np.min(data_4d, axis=(0, 2, 3), keepdims=True)
data_max = np.max(data_4d, axis=(0, 2, 3), keepdims=True)
data_4d = (data_4d - data_min) / (data_max - data_min)

data_4d = torch.from_numpy(data_4d)

torch.save(data_4d, 'paper/data/prep_data/2013_calc_scenario.pt')

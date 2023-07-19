import os
import pickle
import torch


with open('paper/data/prep_data/2013_calc_scenario.pkl', 'rb') as f:
# with open('paper/data/prep_data/2013_Sv05s_LL.pkl', 'rb') as f:
    original_data = pickle.load(f)

with open('paper/data/labels_dictionary.pkl', 'rb') as f:
    labels_dict = pickle.load(f)

images = original_data['images']
labels = original_data['labels']

labels = [label.replace("sd1", "Sv5") for label in labels]

for i in range(10):
    dir_name = 'paper/data/cv_data/crossValidation_all{}'.format(i)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

for i in range(10):
    dir_name = 'paper/data/cv_data/crossValidation_all{}'.format(i)
    for phase in ['train', 'valid', 'test']:
        phase_labels = labels_dict['quakeData-all-crossVaridation{}'.format(i)][phase]
        indexes = [index for index, label in enumerate(labels) if label in phase_labels]
        phase_images = images[indexes]
        phase_images = torch.from_numpy(phase_images)
        torch.save(phase_images, os.path.join(dir_name, 'x_' + phase + '.pt'))
        # torch.save(phase_images, os.path.join(dir_name, 'y_' + phase + '.pt'))

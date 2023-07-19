import torch

data1 = torch.load('paper/data/prep_data/2013_calc_scenario.pt')
data2 = torch.load('paper/data/prep_data/2013_Sv05s_LL.pt')

channels_to_keep = [i for i in range(data1.shape[1]) if i not in [6,7,8,9,10,11]]
channels_to_keep = torch.tensor(channels_to_keep)
data1 = torch.index_select(data1, 1, channels_to_keep)

num_train = 348
num_val = 12

train_data1 = data1[:num_train].clone()
val_data1 = data1[num_train : num_train + num_val].clone()
test_data1 = data1[num_train + num_val :].clone()

train_data2 = data2[:num_train].clone()
val_data2 = data2[num_train : num_train + num_val].clone()
test_data2 = data2[num_train + num_val :].clone()

torch.save(train_data1, 'paper/data/exp_data/x_train.pt')
torch.save(val_data1, 'paper/data/exp_data/x_valid.pt')
torch.save(test_data1, 'paper/data/exp_data/x_test.pt')

torch.save(train_data2, 'paper/data/exp_data/y_train.pt')
torch.save(val_data2, 'paper/data/exp_data/y_valid.pt')
torch.save(test_data2, 'paper/data/exp_data/y_test.pt')

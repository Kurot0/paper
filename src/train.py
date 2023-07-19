import torch
import copy
import yaml
import argparse
import importlib
from ssimloss import SSIMLoss


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class Trainer:
    def __init__(self, params):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.params = params

        data_path = self.params['data_path']
        self.x_train = torch.load(data_path + '/x_train.pt').float().to(self.device)
        self.y_train = torch.load(data_path + '/y_train.pt').float().to(self.device)
        self.x_valid = torch.load(data_path + '/x_valid.pt').float().to(self.device)
        self.y_valid = torch.load(data_path + '/y_valid.pt').float().to(self.device)

        self.checkpoint = self.params['checkpoint_path']

        self.batch_size = self.params['batch_size']
        self.num_epochs = self.params['num_epochs']
        self.patience = self.params['patience']
        self.learning_rate = self.params['learning_rate']

        dataset = torch.utils.data.TensorDataset(self.x_train, self.y_train)
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(self.x_valid, self.y_valid)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        model_module = importlib.import_module(params['model_module'])
        model_class = getattr(model_module, params['model_class'])

        self.model = model_class(**self.params).to(self.device)

        self.criterion = SSIMLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.early_stop_model = None
        self.best_model = None
        self.best_loss = float('inf')

        self.early_stop_flag = self.params['early_stop_flag']

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        for inputs, targets in self.data_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(self.data_loader.dataset)
        return epoch_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
        
        val_loss = running_loss / len(self.val_loader.dataset)
        return val_loss       

    def train(self):
        early_stop = EarlyStopping(patience=self.patience)
        stop_triggered = False
        best_epoch = None
        early_stop_epoch = None
        for epoch in range(self.num_epochs):
            loss = self.train_one_epoch()
            val_loss = self.validate()
            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {-loss}, Validation Loss: {-val_loss}')

            if val_loss < self.best_loss:
                self.best_model = copy.deepcopy(self.model.state_dict())
                self.best_loss = val_loss
                best_epoch = epoch + 1

            early_stop(val_loss)
            if early_stop.early_stop and not stop_triggered:
                print("Early stopping triggered")
                stop_triggered = True
                self.early_stop_model = copy.deepcopy(self.model.state_dict())
                early_stop_epoch = epoch + 1
                if self.early_stop_flag == 1:
                    break

        if self.early_stop_model is not None:
            print(f"The best model : epoch {best_epoch}, Early stopping : epoch {early_stop_epoch}")
        else:
            print(f"The best model : epoch {best_epoch}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='paper/config.yaml', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        params = yaml.safe_load(stream)

    trainer = Trainer(params)
    trainer.train()

    torch.save(trainer.best_model, params['checkpoint_path'] + '/best.pth')
    torch.save(trainer.model.state_dict(), params['checkpoint_path'] + '/last.pth')
    if hasattr(trainer, 'early_stop_model'):
        torch.save(trainer.early_stop_model, params['checkpoint_path'] + '/stop.pth')

if __name__ == '__main__':
    main()

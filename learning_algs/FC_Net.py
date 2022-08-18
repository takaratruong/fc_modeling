import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from typing import Dict
from typing import Tuple
import copy
import time
from mocap.mocap import MoCap
import numpy as np

class Net(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: List[int]):
        super(Net, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()

        dim = input_dim
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(dim, hdim))
            dim = hdim
        self.layers.append(nn.Linear(dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        # out = torch.clip(self.layers[-1](x), -1, 1)
        # out *= 1000
        out = self.layers[-1](x)
        return out

class FC_Net:
    def __init__(self, args):
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.xy_split = args.num_frc

        self.mocap_data = MoCap(args.mocap_path, args)

        self.model = Net(self.mocap_data.obs_size, self.mocap_data.frc_size, args.hidden_dim).to('cuda')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
        self.criterion = nn.MSELoss()

    def train(self):
        min_val_loss = np.inf

        for epoch in range(self.num_epochs):
            train_loss = 0
            val_loss = 0

            train_data, val_data = self.mocap_data.shuffle_split(self.batch_size)

            temp = torch.zeros(0).to('cuda')

            self.model.train()
            for batch in train_data:
                x_trn = batch[:, :-self.xy_split]
                y_trn = batch[:, -self.xy_split:]

                x_trn, y_trn = torch.from_numpy(x_trn).float().to('cuda').type(torch.cuda.FloatTensor), torch.from_numpy(y_trn).float().to('cuda').type(torch.cuda.FloatTensor)

                self.optimizer.zero_grad()
                pred = self.model(x_trn)
                loss = self.criterion(pred, y_trn)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            self.model.eval()
            for batch in val_data:
                x_val = batch[:, :-self.xy_split]
                y_val = batch[:, -self.xy_split:]

                x_val, y_val = torch.from_numpy(x_val).float().to('cuda').type(torch.cuda.FloatTensor), torch.from_numpy(y_val).float().to('cuda').type(torch.cuda.FloatTensor)

                with torch.no_grad():
                    pred = self.model(x_val)

                loss = self.criterion(pred, y_val)
                val_loss += loss.item()
                temp = torch.cat((temp, torch.abs(pred-y_val)))  # delete later

            train_loss = train_loss / len(train_data)
            val_loss = val_loss / len(val_data)

            val_acc = torch.mean(temp)
            val_acc_std = torch.std(temp)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(self.model.state_dict(), 'saved_model.pth')

            if epoch % 500 == 0:
                print('epoch', epoch, '\t Training Loss: %.4f' % train_loss, ' \t Validation Loss: %.4f' % val_loss, ' Validation Acc: %.3f' % val_acc, '+- %.3f' % val_acc_std)

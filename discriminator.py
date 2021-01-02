"""
Define the discriminator model for the GAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_gen import generate_dataset

def binary_accuracy(y_pred, y_true):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_true).sum().float()
    acc = correct_results_sum/y_true.shape[0]
    acc = torch.round(acc * 100)
    return acc

def train_discriminator(data: DataLoader, lr=0.01, epochs=10):
    model = Discriminator()
    optimiser = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        for x, y in data:
            optimiser.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            acc = binary_accuracy(output, y)
            optimiser.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        print(f'Epoch {epoch} loss: {epoch_loss/len(data)}, '
            f'acc: {epoch_acc/len(data)}')
    
    return model

class Discriminator(nn.Module):
    """
    Simple feed-forward NN to discriminate the 'real' and generated data
    """
    def __init__(self):
        super().__init__()
        ## one hidden layer with 25 nodes
        self.hidden_1 = nn.Linear(2, 25)
        self.output_layer = nn.Linear(25, 1)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.hidden_1(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x

if __name__ == '__main__':
    dataset = generate_dataset(128)
    loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

    train_discriminator(loader)
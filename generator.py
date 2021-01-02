"""
Code to define the generator network
"""

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # single hidden layer with 15 nodes
        self.hidden_1 = nn.Linear(latent_dim, 15)
        self.output_layer = nn.Linear(15, 2)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.hidden_1(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x

def test_generator(data):
    latent_dim = data.shape[1]
    gen = Generator(latent_dim)
    gen.eval()
    output = gen(data)
    print(output)
    return


if __name__ == '__main__':
    latent_dim = 10
    n_points = 10
    data = torch.randn(latent_dim * n_points)
    data = torch.reshape(data, [n_points, latent_dim])
    print(data)
    test_generator(data)

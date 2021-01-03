"""
Code to define the generator network
"""

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, n_hn=15):
        super().__init__()
        # single hidden layer with 15 nodes
        self.hidden_1 = nn.Linear(latent_dim, n_hn)
        self.output_layer = nn.Linear(n_hn, 2)
        self.relu = nn.LeakyReLU()

        self.main = nn.Sequential(
            self.hidden_1,
            self.relu,
            self.output_layer
        )
    
    def forward(self, x):
        x = self.main(x)
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

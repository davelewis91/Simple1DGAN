"""
Code to train a simple 1D GAN
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from discriminator import Discriminator
from generator import Generator
from data_gen import generate_dataset

REAL_LABEL = 1
FAKE_LABEL = 0

def train_gan(data: DataLoader, latent_dim: int=10, n_epochs: int=10, lr: float=0.01):
    gen_model = Generator(latent_dim)
    dis_model = Discriminator()

    gen_optim = optim.Adam(gen_model.parameters(), lr=lr)
    dis_optim = optim.Adam(dis_model.parameters(), lr=lr)

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        d_epoch_loss = 0
        g_epoch_loss = 0
        for x, y in data:
            # forward pass through discriminator on only-real data
            dis_model.zero_grad()
            output = dis_model(x).view(-1)
            loss_real = criterion(output, y)
            loss_real.backward()

            # forward pass with fake data
            noise = torch.randn(x.shape[0], latent_dim)
            fake_x = gen_model(noise)
            label = torch.Tensor([FAKE_LABEL]*x.shape[0])
            output = dis_model(fake_x.detach()).view(-1)
            loss_fake = criterion(output, label)
            loss_fake.backward()
            
            # combine real and fake losses
            loss = loss_real + loss_fake
            d_epoch_loss += loss.item()
            
            # update discriminator
            dis_optim.step()

            # now update generator based on discriminator performance
            gen_model.zero_grad()
            label = torch.Tensor([REAL_LABEL]*x.shape[0])
            ## discriminator updated, so check how it's doing now
            output = dis_model(fake_x).view(-1)
            gen_loss = criterion(output, label)
            gen_loss.backward()
            gen_optim.step()
            g_epoch_loss += gen_loss.item()
        
        print(f'Epoch {epoch} DLoss: {d_epoch_loss/(len(data)*2)}, '
            f'GLoss: {g_epoch_loss/len(data)}')

    return


if __name__ == '__main__':
    from data_gen import TrainingSet, generate_points

    n_points = 128
    data = generate_points(n_points, power=2)
    target = torch.ones(n_points)
    #target = torch.reshape(target, [-1, 1])
    dataset = TrainingSet(data, target)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    train_gan(loader)

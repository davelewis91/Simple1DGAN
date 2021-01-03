"""
Code to train a simple 1D GAN
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from discriminator import Discriminator
from generator import Generator
from data_gen import generate_dataset

REAL_LABEL = 1
FAKE_LABEL = 0

def update_model(model, criterion, x, y):
    output = model(x).view(-1)
    loss = criterion(output, y)
    loss.backward()
    return output, loss

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.2)
        m.bias.data.fill_(0.1)

def make_noise(n_points, latent_dim):
    return torch.rand(n_points, latent_dim) * 2 - 1

def train_gan(data: DataLoader,
              latent_dim: int = 10,
              n_dis_hn: int = 25,
              n_gen_hn: int = 15,
              n_epochs: int = 10,
              lr: float = 0.01, 
              early_stop: bool = False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    gen_model = Generator(latent_dim, n_gen_hn).to(device)
    gen_model.apply(init_weights)
    dis_model = Discriminator(n_dis_hn).to(device)
    dis_model.apply(init_weights)

    gen_optim = optim.Adam(gen_model.parameters(), lr=lr)
    dis_optim = optim.Adam(dis_model.parameters(), lr=lr)

    criterion = nn.BCELoss()

    d_losses = []
    g_losses = []
    D_vals = []

    prev_loss = 0
    prev_loss_counter = 0
    min_gen_loss = 100
    best_epoch = 0
    stopped = False
    for epoch in range(n_epochs):
        d_epoch_loss = 0
        g_epoch_loss = 0
        iter = 0  # count batches per epoch
        for x, y in data:
            # forward pass through discriminator on only-real data
            dis_model.zero_grad()
            output, loss_real = update_model(dis_model, criterion, x, y)
            D_x = output.mean().item()

            # forward pass with fake data
            noise = make_noise(x.shape[0], latent_dim)
            fake_x = gen_model(noise)
            label = torch.Tensor([FAKE_LABEL]*x.shape[0])
            output, loss_fake = update_model(
                dis_model, 
                criterion, 
                fake_x.detach(), 
                label
            )
            
            # combine real and fake losses
            loss = loss_real + loss_fake
            d_epoch_loss += loss.item()
            D_g_z1 = output.mean().item()
            
            # update discriminator
            dis_optim.step()

            # now update generator based on discriminator performance
            gen_model.zero_grad()
            label = torch.Tensor([REAL_LABEL]*x.shape[0])
            ## discriminator updated, so check how it's doing now
            output, gen_loss = update_model(dis_model, criterion, fake_x, label)
            gen_optim.step()
            g_epoch_loss += gen_loss.item()
            D_g_z2 = output.mean().item()

            d_losses.append(loss.item())
            g_losses.append(gen_loss.item())
            D_vals.append([D_x, D_g_z1, D_g_z2])

            iter += 1

        if epoch % 100 == 0:
            print(f'Epoch {epoch} DLoss: {d_epoch_loss}, GLoss: {g_epoch_loss}')

        # early stopping
        if early_stop:
            if g_epoch_loss < min_gen_loss and epoch > 0:
                min_gen_loss = g_epoch_loss
                best_epoch = epoch
                torch.save(gen_model, 'best_gen_model.torch')
            elif epoch - best_epoch > 1000:
                print(f'Model not learning, {epoch - best_epoch} epochs '
                    f'since best epoch {best_epoch} ({min_gen_loss})')
                stopped = True
                break
            elif g_epoch_loss > 500:
                print(f'Sudden loss explosion: E{epoch}: {g_epoch_loss}')
                stopped = True
                break

            # handle when discriminator loss is not changing
            if round(d_epoch_loss, 3) != prev_loss:
                prev_loss = round(d_epoch_loss, 3)
                prev_loss_counter = 0
            else:
                prev_loss_counter += 1
            if prev_loss_counter > 10:
                print(f'Model converged, epoch {epoch}')
                break
    
    plt.plot(d_losses, label='discriminator', alpha=0.6)
    plt.plot(g_losses, label='generator', alpha=0.6)
    plt.vlines(best_epoch * iter, 0, plt.ylim()[1], color='red')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.savefig('../MyHomeFolder/training_losses.png', bbox_inches='tight')
    plt.clf()

    plt.plot([x[0] for x in D_vals], label='D_x', alpha=0.6)
    plt.plot([x[1] for x in D_vals], label='D_g_z1', alpha=0.6)
    plt.plot([x[2] for x in D_vals], label='D_g_z2', alpha=0.6)
    plt.ylim(-0.1, 1.1)
    plt.hlines(0.5, 0, plt.xlim()[1], color='red')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('D(x)')
    plt.savefig('../MyHomeFolder/d_values.png', bbox_inches='tight')
    plt.clf()

    if stopped:
        gen_model = torch.load('best_gen_model.torch')

    return gen_model, d_losses, g_losses


if __name__ == '__main__':
    from data_gen import TrainingSet, generate_points

    #seed = 4321
    #torch.manual_seed(seed)

    n_points = 200
    latent_dim = 5
    n_epochs = 1000
    n_gen_hn = 10
    n_dis_hn = 20

    eval_latent_dim = False

    data = generate_points(n_points, 'sin')
    target = torch.ones(n_points)
    dataset = TrainingSet(data, target)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    if eval_latent_dim:
        d_losses = []
        g_losses = []
        for i in range(2, 14):
            gan, d_loss, g_loss = train_gan(
                loader, 
                latent_dim=i,
                n_dis_hn=n_dis_hn,
                n_gen_hn=n_gen_hn,
                n_epochs=n_epochs, 
                early_stop=False
            )
            d_losses.append(d_loss)
            g_losses.append(g_loss)
        for i, loss in enumerate(d_losses):
            plt.plot(loss, label=i+2)
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('discriminator loss')
        plt.savefig('../MyHomeFolder/latent_dim_dis_losses.png', bbox_inches='tight')
        plt.clf()
        for i, loss in enumerate(g_losses):
            plt.plot(loss, label=i+2)
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('generator loss')
        plt.savefig('../MyHomeFolder/latent_dim_gen_losses.png', bbox_inches='tight')
        plt.clf()
    else:
        gan, d_loss, g_loss = train_gan(
            loader, 
            latent_dim=latent_dim, 
            n_dis_hn=n_dis_hn,
            n_gen_hn=n_gen_hn,
            n_epochs=n_epochs, 
            early_stop=False
        )

        plt.scatter(data[:,0], data[:,1], label='real', alpha=0.6)
        noise = make_noise(n_points, latent_dim)
        with torch.no_grad():
            fake_data = gan(noise).detach()
        plt.scatter(fake_data[:,0], fake_data[:,1], label='fake', alpha=0.6)
        plt.legend()
        plt.savefig('../MyHomeFolder/test_plot.png', bbox_inches='tight')
        plt.clf()



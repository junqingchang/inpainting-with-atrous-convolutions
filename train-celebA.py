import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from celeba import CelebA
from atrousinpainter import AtrousInpainter, Discriminator
import os
import matplotlib.pyplot as plt


data_dir = 'data/CelebA'
model_dir = 'celebAchkpt/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
plots_dir = 'celebAplots/'
if not os.path.exists(plots_dir):
    os.mkdir(plots_dir)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0001
SAVE_EVERY = 10
PLOT_EVERY = 10


def visualise_progress(train_data, model, device, epoch):
    model.eval()
    for i in range(3):
        data, target = train_data[i]
        data = data.unsqueeze(0).to(device)
        reconstructed_img = model(data).squeeze()
        plt.subplot(3, 3, i*3+1)
        plt.imshow(data.detach().squeeze().cpu().transpose(0, 1).transpose(1, 2))
        plt.subplot(3, 3, i*3+2)
        plt.imshow(target.transpose(0, 1).transpose(1, 2))
        plt.subplot(3, 3, i*3+3)
        plt.imshow(reconstructed_img.detach().cpu().transpose(0, 1).transpose(1, 2))
    plt.savefig(os.path.join(plots_dir, f'epoch{epoch}progress'))
    plt.close()



def train(train_loader, model, discrim, optimizer_G, optimizer_D, device, criterion_R, criterion_G, print_every=1000):
    model.train()
    discrim.train()
    total_r_loss = 0
    total_d_loss = 0
    total_g_loss = 0
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        ## Recon
        reconstructed_img = model(data)
        reconstruction_loss = criterion_R(reconstructed_img, target)
        total_r_loss += reconstruction_loss.item()
        reconstruction_loss.backward()
        optimizer_G.step()

        ## Discrim
        model.eval()
        reconstructed_img = model(data)

        real_label = torch.ones((BATCH_SIZE, 1)).to(device)
        fake_label = torch.zeros((BATCH_SIZE, 1)).to(device)

        discrim_real = discrim(target)
        discrim_fake = discrim(reconstructed_img)
        real_loss = criterion_G(discrim_real, real_label)
        fake_loss = criterion_G(discrim_fake, fake_label)

        discriminator_loss = real_loss + fake_loss
        total_d_loss += discriminator_loss.item()
        discriminator_loss.backward()
        optimizer_D.step()

        ## Generation
        model.train()
        reconstructed_img = model(data)
        discrim_train = discrim(reconstructed_img)
        
        generation_loss = criterion_G(discrim_train, real_label)
        total_g_loss += generation_loss.item()
        generation_loss.backward()
        optimizer_G.step()
        
        if i % print_every == 0:
            print(f'{i}/{len(train_loader)} R_Loss: {reconstruction_loss.item()}, D_Loss: {discriminator_loss.item()}, G_Loss: {generation_loss.item()}')

    return total_r_loss/len(train_loader), total_d_loss/len(train_loader), total_g_loss/len(train_loader)


if __name__ == '__main__':
    train_data = CelebA(data_dir)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    model = AtrousInpainter()
    model.to(device)

    discrim = Discriminator()
    discrim.to(device)

    criterion_R = nn.MSELoss()
    criterion_G = nn.BCELoss()

    optimizer_G = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer_D = optim.Adam(discrim.parameters(), lr=LEARNING_RATE)

    r_losses = []
    d_losses = []
    g_losses = []

    for epoch in range(1, EPOCHS+1):
        r_loss, d_loss, g_loss = train(train_loader, model, discrim, optimizer_G, optimizer_D, device, criterion_R, criterion_G)
        r_losses.append(r_loss)
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        print()
        print(f'Epoch {epoch}: R_Loss: {r_loss}, D_Loss: {d_loss}, G_Loss: {g_loss}')

        plt.title('Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(r_losses)
        plt.savefig(os.path.join(plots_dir, 'r_loss.png'))
        plt.close()

        plt.title('Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(d_losses)
        plt.savefig(os.path.join(plots_dir, 'd_loss.png'))
        plt.close()

        plt.title('Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(g_losses)
        plt.savefig(os.path.join(plots_dir, 'g_loss.png'))
        plt.close()

        if epoch % SAVE_EVERY == 0:
            torch.save(model, os.path.join(model_dir, f'generator-epoch{epoch}.pt'))
            torch.save(discrim, os.path.join(model_dir, f'discriminator-epoch{epoch}.pt'))

        if epoch % PLOT_EVERY == 0:
            visualise_progress(train_data, model, device, epoch)
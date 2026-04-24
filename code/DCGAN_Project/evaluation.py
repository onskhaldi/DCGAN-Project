import torch
from torch import nn
import numpy as np

def evaluate(D, G, G_latent_dim, test_loader):
    D.eval()
    G.eval()
    device = next(D.parameters()).device

    D_batch_losses = []
    G_batch_losses = []

    with torch.no_grad():
        for batch_imgs, _ in test_loader:
            batch_imgs = batch_imgs.to(device)

            real_labels = torch.ones((batch_imgs.size(0), 1), device=device)
            fake_labels = torch.zeros((batch_imgs.size(0), 1), device=device)

            # Discriminator real loss
            D_real_output = D(batch_imgs)
            D_real_loss = nn.functional.binary_cross_entropy_with_logits(D_real_output, real_labels)

            # Discriminator fake loss
            z = torch.randn(batch_imgs.size(0), G_latent_dim, 1, 1, device=device)
            fake_imgs = G(z)
            D_fake_output = D(fake_imgs)
            D_fake_loss = nn.functional.binary_cross_entropy_with_logits(D_fake_output, fake_labels)

            D_loss = D_real_loss + D_fake_loss
            D_batch_losses.append(D_loss.item())

            # Generator loss
            z = torch.randn(batch_imgs.size(0), G_latent_dim, 1, 1, device=device)
            fake_imgs = G(z)
            D_output_on_fake = D(fake_imgs)
            G_loss = nn.functional.binary_cross_entropy_with_logits(D_output_on_fake, real_labels)
            G_batch_losses.append(G_loss.item())

    D.train()
    G.train()

    return np.mean(D_batch_losses), np.mean(G_batch_losses)

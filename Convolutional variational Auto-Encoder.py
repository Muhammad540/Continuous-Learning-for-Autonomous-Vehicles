import torch
import torch.nn as nn
class Convolution_Variational_Autoencoder(nn.Module):
    def __init__(self):
        super(Convolution_Variational_Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),  # Input: 3x800x800, Output: 16x400x400
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Output: 16x200x200
            nn.Conv2d(16, 32, 4, 2, 1),  # Output: 32x100x100
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Output: 32x50x50
            nn.Conv2d(32, 64, 4, 2, 1),  # Output: 64x25x25
            nn.ReLU(True)
        )
        # Latent space
        self.fc_mu = nn.Linear(64 * 25 * 25, 256)  # Output: 256
        self.fc_logvar = nn.Linear(64 * 25 * 25, 256)  # Output: 256

        # Decoder
        self.decoder_fc = nn.Linear(256, 64 * 25 * 25)  # Input: 256, Output: Flattened 64x25x25

        self.decoder = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # Input: 64x25x25, Output: 32x50x50
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # Output: 32x100x100
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # Output: 16x200x200
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, 4, 2, 1),  # Output: 16x400x400
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),  # Output: 3x800x800
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        epsilon = torch.randn(mu.size(0), mu.size(1))             #.to(mu.device)  # Ensuring epsilon is on the same device as mu
        z = mu + epsilon * torch.exp(logvar / 2)  # z = mean + std * epsilon
        return z

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the output from the encoder

        # Latent space
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        # Decoder
        z = self.decoder_fc(z)  # Map z to the initial shape for the decoder
        z = z.view(z.size(0), 64, 25, 25)  # Reshape z to the expected input dimensions of the decoder
        x_recon = self.decoder(z)  # Pass z through the decoder to get the reconstruction

        return x_recon, mu, logvar

#### ALL THE IMPORTS ####
import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms import v2
from torch.utils.data import Dataset
from torchvision.io import read_image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from .CVAE import Convolution_Variational_Autoencoder # Import the model
from .Datasetloader import SingleImageDataset # Import the custom dataset
# Lets convert this code into a function
# ----------------------- Inference -----------------------
def inference(img):
    """
    Perform inference on a single image using a pre-trained Convolutional Autoencoder.

    Args:
        img (str or PIL.Image): Path to the image or the image object itself.

    Returns:
        float, float: Mean and standard deviation of the reconstruction error.
    """

    #weights_file = r'D:\Carla_new\WindowsNoEditor\PythonAPI\examples\new_env\CAE\weights\conv_vae_epoch_50_old.pth'
    weights_file = r'D:\Carla_new\WindowsNoEditor\PythonAPI\examples\new_env\CAE\weights\conv_vae.pth'
    if not os.path.isfile(weights_file):
        raise FileNotFoundError(f"Weights File not found: {weights_file}")

    # Load model
    loaded_model = Convolution_Variational_Autoencoder()
    loaded_model.load_state_dict(torch.load(weights_file, map_location=torch.device('cpu')))
    loaded_model.eval()

    # Define transforms
    transform = transforms.Compose([transforms.ToTensor()])

    # Initialize data loader
    data_loader = DataLoader(SingleImageDataset(img, transform=transform), batch_size=1, shuffle=False)

    # Calculate the combined loss of the incoming image '
    with torch.no_grad():
        for (img,_) in data_loader:
            x_recon, mu, log_sigma_sq = loaded_model(img)
            # Calculate reconstruction loss
            recon_loss = torch.nn.functional.mse_loss(x_recon, img)
            # Calculate KL divergence
            kl_divergence_loss = -0.5 * torch.sum(1 + log_sigma_sq - mu.pow(2) - (log_sigma_sq.exp() + 1e-8), axis=1)
            # Calculate combined loss
            combined_loss = recon_loss + kl_divergence_loss

    return combined_loss.item()

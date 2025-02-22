import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from data.dataset import ReconstructionDataset
from models.unrolled_network import UnrolledNetwork
from models.unet import LogAleatoricVarianceNetwork
import utils.train_utils as train_utils

import matplotlib.pyplot as plt

# --------------------- Load and Print Configuration ---------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

print("Loaded Configurations:")
print(yaml.dump(config, default_flow_style=False))

# Extract configuration parameters

lambda_init = config["model"]["lambda_init"]
num_iter = config["model"]["num_iter"]
dropout_rate = config["model"]["dropout_rate"]
n_channels = config["model"]["n_channels"]

residual_block_kwargs = config["model"]["residual_block_kwargs"]

log_aleatoric_variance_network_config = config["model"]["log_aleatoric_variance_network"]
log_aleatoric_variance_network_p = log_aleatoric_variance_network_config["p"]
log_aleatoric_variance_network_num_levels = log_aleatoric_variance_network_config["num_levels"]
log_aleatoric_variance_network_base_channels = log_aleatoric_variance_network_config["base_channels"]

test_npz_file_dir = config["paths"]["test_npz_file_dir"]

checkpoint_path = config["test"]["test_checkpoint_path"]

T = config["test"]["number_of_mc_dropout_passes"]

# Fix the random seed and obtain the device info
device = train_utils.init(seed=42)
print("Using the device:", device)

# --------------------- Dataset and Dataloaders ---------------------
test_dataset = ReconstructionDataset(npz_file=test_npz_file_dir)
print("Size of the test dataset:", len(test_dataset))

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# --------------------- Forward and Adjoint Operators ---------------------
mask = torch.from_numpy(np.load(test_npz_file_dir)["mask"]).float().to(device)  # Expected shape: (256,256)

def A(x):
    # Convert two-channel input to complex representation.
    x_complex = x[:, 0, :, :] + 1j * x[:, 1, :, :]
    out = torch.fft.fft2(x_complex, norm="ortho")
    out = out * mask  # Apply subsampling mask.
    out_real = torch.real(out)
    out_imag = torch.imag(out)
    return torch.stack((out_real, out_imag), dim=1)

def A_adjoint(x):
    # Convert two-channel input to complex representation.
    x_complex = x[:, 0, :, :] + 1j * x[:, 1, :, :]
    out = x_complex * mask  # Apply mask.
    out = torch.fft.ifft2(out, norm="ortho")
    out_real = torch.real(out)
    out_imag = torch.imag(out)
    return torch.stack((out_real, out_imag), dim=1)

# --------------------- Models ---------------------
unrolled_network = UnrolledNetwork(
    A=A,
    A_adjoint=A_adjoint,
    lamb=lambda_init,
    num_iter=num_iter,
    residual_block_kwargs=residual_block_kwargs
).to(device)

log_aleatoric_variance_network = LogAleatoricVarianceNetwork(
    n_channels=n_channels,
    base_channels=log_aleatoric_variance_network_base_channels,
    num_levels=log_aleatoric_variance_network_num_levels,
    p=log_aleatoric_variance_network_p
).to(device)

# --------------------- Test ---------------------
unrolled_network.eval()
log_aleatoric_variance_network.eval()

for i, data in enumerate(test_dataloader):
    measurement, gt = data
    measurement = measurement.to(device)
    gt = gt.to(device)

    means = []
    logvars = []

    # Feed the measurement to the network T times
    for t in range(T):
        with torch.no_grad():
            starting_point = A_adjoint(measurement)
            unrolled_network_outputs = unrolled_network(measurement, starting_point)
            log_aleatoric_variance_network_outputs = log_aleatoric_variance_network(starting_point)
            # Append the lists
            means.append(unrolled_network_outputs.detach().cpu().numpy())
            logvars.append(log_aleatoric_variance_network_outputs.detach().cpu().numpy())

    # Convert the lists to numpy arrays
    means = np.array(means) # [T,1,2,H,W]
    logvars = np.array(logvars) # [T,1,2,H,W]

    # Get the ground truth image
    gt = gt.detach().cpu().numpy()
    complex_gt = gt[:,0,:,:] + 1j * gt[:,1,:,:] # [1,H,W]
    complex_gt = np.squeeze(complex_gt, 0)

    # Get the zero-filled reconstruction
    starting_point = starting_point.detach().cpu().numpy()
    complex_starting_point = starting_point[:,0,:,:] + 1j * starting_point[:,1,:,:] # [1,H,W]
    complex_starting_point = np.squeeze(complex_starting_point, 0)

    # Calculate the predictive mean
    complex_means = means[:,:,0,:,:] + 1j * means[:,:,1,:,:] # [T,1,H,W]
    pred_mean = np.squeeze(np.mean(complex_means, 0), 0)

    # Calculate the epistemic uncertainty
    complex_means = means[:,:,0,:,:] + 1j * means[:,:,1,:,:] # [T,1,H,W]
    epistemic_std = np.squeeze(np.std(complex_means, 0), 0)

    # Calculate the aleatoric uncertainty
    vars = np.exp(logvars) # [T,1,2,H,W]
    complex_vars = vars[:,:,0,:,:] + vars[:,:,1,:,:] # [T,1,H,W]
    aleatoric_std = np.squeeze(np.mean(complex_vars, 0), 0)**0.5

    # calculate the total uncertainty
    pred_uncertainty_std = (epistemic_std**2 + aleatoric_std**2)**0.5

    # Display everything
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    axs[0,0].imshow(np.abs(complex_gt), cmap='gray')
    axs[0,0].set_title("Ground Truth Magnitude")
    axs[0,0].axis("off")

    axs[0,1].imshow(np.angle(complex_gt), cmap='gray')
    axs[0,1].set_title("Ground Truth Phase")
    axs[0,1].axis("off")

    axs[0,2].imshow(mask.detach().cpu().numpy(), cmap='gray')
    axs[0,2].set_title("Mask")
    axs[0,2].axis("off")

    axs[0,3].imshow(np.abs(complex_starting_point), cmap='gray')
    axs[0,3].set_title("Zero-Filling Magnitude")
    axs[0,3].axis("off")

    axs[0,4].imshow(np.angle(complex_starting_point), cmap='gray')
    axs[0,4].set_title("Zero-Filling Phase")
    axs[0,4].axis("off")

    axs[1,0].imshow(np.abs(pred_mean), cmap='gray')
    axs[1,0].set_title("Zero-Filling Magnitude")
    axs[1,0].axis("off")

    axs[1,1].imshow(np.angle(pred_mean), cmap='gray')
    axs[1,1].set_title("Zero-Filling Phase")
    axs[1,1].axis("off")

    axs[1,2].imshow(epistemic_std, cmap='jet')
    axs[1,2].set_title("Epistemic Uncertainty")
    axs[1,2].axis("off")

    axs[1,3].imshow(aleatoric_std, cmap='jet')
    axs[1,3].set_title("Aleatoric Uncertainty")
    axs[1,3].axis("off")

    axs[1,4].imshow(pred_uncertainty_std, cmap='jet')
    axs[1,4].set_title("Predictive (Total) Uncertainty")
    axs[1,4].axis("off")
    plt.show()

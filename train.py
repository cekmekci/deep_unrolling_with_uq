import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from data.dataset import ReconstructionDataset
from models.unrolled_network import UnrolledNetwork
from models.unet import LogAleatoricVarianceNetwork
import utils.train_utils as train_utils

# --------------------- Load and Print Configuration ---------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

print("Loaded Configuration:")
print(yaml.dump(config, default_flow_style=False))

# Extract configuration parameters
train_npz_file_dir = config["paths"]["train_npz_file_dir"]
val_npz_file_dir = config["paths"]["val_npz_file_dir"]
checkpoint_dir = config["paths"]["checkpoint_dir"]

train_batch_size = config["training"]["train_batch_size"]
val_batch_size = config["training"]["val_batch_size"]
lr = config["training"]["lr"]
num_epochs = config["training"]["num_epochs"]
checkpoint_save_interval = config["training"]["checkpoint_save_interval"]

lambda_init = config["model"]["lambda_init"]
num_iter = config["model"]["num_iter"]
dropout_rate = config["model"]["dropout_rate"]
n_channels = config["model"]["n_channels"]

residual_block_kwargs = config["model"]["residual_block_kwargs"]

log_aleatoric_variance_network_config = config["model"]["log_aleatoric_variance_network"]
log_aleatoric_variance_network_p = log_aleatoric_variance_network_config["p"]
log_aleatoric_variance_network_num_levels = log_aleatoric_variance_network_config["num_levels"]
log_aleatoric_variance_network_base_channels = log_aleatoric_variance_network_config["base_channels"]

# Fix the random seed and obtain the device info
device = train_utils.init(seed=42)
print("Using the device:", device)

# Create the checkpoint directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)

# --------------------- Dataset and Dataloaders ---------------------
train_dataset = ReconstructionDataset(npz_file=train_npz_file_dir)
val_dataset = ReconstructionDataset(npz_file=val_npz_file_dir)
print("Size of the training dataset:", len(train_dataset))
print("Size of the validation dataset:", len(val_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

# --------------------- Forward and Adjoint Operators ---------------------
mask = torch.from_numpy(np.load(train_npz_file_dir)["mask"]).float().to(device)  # Expected shape: (256,256)

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

# --------------------- Loss and Optimizers ---------------------
criterion = train_utils.kendall_loss_function

weight_decay = (1 - dropout_rate) / (2 * len(train_dataset))
optimizer1 = torch.optim.Adam(unrolled_network.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(log_aleatoric_variance_network.parameters(), lr=lr, weight_decay=weight_decay)

best_val_loss = float("inf")

# --------------------- Training Loop ---------------------
for epoch in range(num_epochs):

    # Training
    running_train_loss = 0.0
    unrolled_network.train()
    log_aleatoric_variance_network.train()
    for i, data in enumerate(train_dataloader):
        measurement, gt = data
        measurement = measurement.to(device)
        gt = gt.to(device)

        starting_point = A_adjoint(measurement)
        unrolled_network_outputs = unrolled_network(measurement, starting_point)
        log_aleatoric_variance_network_outputs = log_aleatoric_variance_network(starting_point)

        loss = criterion(gt, unrolled_network_outputs, log_aleatoric_variance_network_outputs)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()

        running_train_loss += loss.item()
    running_train_loss /= len(train_dataloader)

    # Validation
    running_val_loss = 0.0
    unrolled_network.eval()
    log_aleatoric_variance_network.eval()
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            measurement, gt = data
            measurement = measurement.to(device)
            gt = gt.to(device)
            starting_point = A_adjoint(measurement)
            unrolled_network_outputs = unrolled_network(measurement, starting_point)
            log_aleatoric_variance_network_outputs = log_aleatoric_variance_network(starting_point)
            loss = criterion(gt, unrolled_network_outputs, log_aleatoric_variance_network_outputs)
            running_val_loss += loss.item()
    running_val_loss /= len(val_dataloader)

    # Print the losses
    print('Epoch [{}/{}], Training Loss: {:.8f} Validation Loss: {:.8f}'
          .format(epoch + 1, num_epochs, running_train_loss, running_val_loss))

    # Save the models
    if epoch % checkpoint_save_interval == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        checkpoint = {
            "epoch": epoch,
            "unrolled_network_state_dict": unrolled_network.state_dict(),
            "log_aleatoric_variance_network_state_dict": log_aleatoric_variance_network.state_dict(),
            "optimizer_unrolled_network_state_dict": optimizer1.state_dict(),
            "optimizer_log_aleatoric_variance_network_state_dict": optimizer2.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

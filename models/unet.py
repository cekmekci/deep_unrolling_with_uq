import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import MCDropout


class DoubleConv(nn.Module):
    """Double convolution block using Conv2d, Dropout, BatchNorm, and ReLU.

    This module applies two consecutive sets of operations, each consisting of
    a convolution followed by dropout, batch normalization, and a ReLU activation.
    An optional intermediate channel size can be provided; otherwise, it defaults
    to the number of output channels.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mid_channels (int, optional): Number of channels in the intermediate layer.
            Defaults to out_channels if not specified.
        p (float): Dropout rate
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, p=0.1):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            MCDropout(p),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            MCDropout(p),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Forward pass through the double convolution block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying two consecutive
                convolutional operations.
        """
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block: Max pooling followed by a double convolution.

    This module first reduces the spatial dimensions using MaxPool2d and then
    applies a DoubleConv block to extract features.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        p (float): Dropout rate
    """
    def __init__(self, in_channels, out_channels, p=0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, p=p)
        )

    def forward(self, x):
        """Forward pass through the downscaling block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after max pooling and double convolution.
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block followed by a double convolution.

    This module upsamples the input tensor via bilinear interpolation, pads it
    if necessary to match dimensions, concatenates it with a corresponding
    encoder feature map (skip connection), and then applies a DoubleConv block.

    Args:
        in_channels (int): Number of channels from the concatenated input.
        out_channels (int): Number of output channels.
        p (float): Dropout rate
    """
    def __init__(self, in_channels, out_channels, p=0.1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, p=p)

    def forward(self, x1, x2):
        """Forward pass through the upscaling block.

        Upsamples the first input tensor, pads it to match the dimensions of the
        second input tensor (skip connection), concatenates them, and applies a
        double convolution.

        Args:
            x1 (torch.Tensor): The tensor to be upsampled.
            x2 (torch.Tensor): The corresponding tensor from the encoder
                (skip connection).

        Returns:
            torch.Tensor: Output tensor after upsampling, concatenation, and
                convolution.
        """
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution to map the feature maps to the desired output channels.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass through the output convolution.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the 1x1 convolution.
        """
        return self.conv(x)


class LogAleatoricVarianceNetwork(nn.Module):
    """Configurable U-Net style architecture for modeling aleatoric uncertainty.

    This network predicts the logarithm of the aleatoric variance for a given input.
    It consists of an encoder (downsampling path) and a decoder (upsampling path) with
    skip connections, similar to U-Net architectures.

    Args:
        n_channels (int): Number of input channels.
        base_channels (int, optional): Number of filters for the initial convolutional layer.
            Defaults to 64.
        num_levels (int, optional): Number of downsampling levels (excluding the initial layer),
            which determines the depth of the network. Defaults to 4.
        p (float): Dropout rate
    """
    def __init__(self, n_channels, base_channels=64, num_levels=4, p=0.1):
        super().__init__()
        self.num_levels = num_levels
        self.base_channels = base_channels

        # Compute encoder channel sizes for each level.
        self.enc_channels = [base_channels * (2 ** i) for i in range(num_levels)]
        # Compute the bottom layer channel size based on the upsampling strategy.
        self.bottom_channels = base_channels * (2 ** (num_levels - 1))


        # Initial convolution block.
        self.inc = DoubleConv(n_channels, self.enc_channels[0], p=p)

        # Build the encoder (Down blocks).
        self.down_blocks = nn.ModuleList()
        in_ch = self.enc_channels[0]
        for i in range(1, num_levels):
            out_ch = self.enc_channels[i]
            self.down_blocks.append(Down(in_ch, out_ch, p=p))
            in_ch = out_ch
        # Final Down block for the bottom layer.
        self.down_blocks.append(Down(in_ch, self.bottom_channels, p=p))

        # Build the decoder (Up blocks).
        self.up_blocks = nn.ModuleList()
        current_channels = self.bottom_channels
        # Iterate through encoder levels in reverse order for skip connections.
        for i in range(num_levels - 1, -1, -1):
            skip_ch = self.enc_channels[i]
            in_channels = current_channels + skip_ch
            out_channels = skip_ch // 2 if i > 0 else skip_ch
            self.up_blocks.append(Up(in_channels, out_channels, p=p))
            current_channels = out_channels

        # Final output convolution.
        self.outc = OutConv(self.enc_channels[0], n_channels)

    def forward(self, x):
        """Forward pass through the LogAleatoricVarianceNetwork network.

        The input tensor passes through the encoder while storing skip
        connection features. The decoder upsamples the features and concatenates
        them with the corresponding encoder outputs. Finally, a 1x1 convolution
        produces the predicted log aleatoric variance.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor representing the predicted logarithm of
            the aleatoric variance.
        """
        x_skips = []
        x = self.inc(x)
        x_skips.append(x)
        for i in range(len(self.down_blocks) - 1):
            x = self.down_blocks[i](x)
            x_skips.append(x)
        x = self.down_blocks[-1](x)
        for up, skip in zip(self.up_blocks, reversed(x_skips)):
            x = up(x, skip)
        x = self.outc(x)
        return x


# Test the model implementation
if __name__ == '__main__':
    # Import the torchsummary package for a detailed model summary.
    from torchsummary import summary

    # Instantiate the ProposedLogvar network.
    # Here, we assume an input with 3 channels (e.g., RGB image).
    model = LogAleatoricVarianceNetwork(n_channels=3, base_channels=64, num_levels=4, p=0.1)

    # Print the model summary.
    # The input size is specified as (channels, height, width).
    summary(model, input_size=(3, 256, 256))

    # Create a random input tensor with shape (batch_size, channels, height, width).
    input_tensor = torch.randn(1, 3, 256, 256)

    # Pass the input through the network.
    output = model(input_tensor)
    print("Output shape:", output.shape)

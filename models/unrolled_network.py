import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import MCDropout


class ResidualBlock(nn.Module):
    """Residual block in the ResNet architecture.

    This block applies a sequence of convolutional layers, each followed by dropout
    and (except for the final layer) a LeakyReLU activation. The final convolutional
    layer must have the same number of filters as the input channels to allow for an
    identity skip connection.

    Args:
        num_filter_list (list[int]): Number of filters for each convolutional layer.
        kernel_size_list (list[int] or list[tuple]): Kernel sizes for each convolutional layer.
        stride_list (list[int] or list[tuple]): Strides for each convolutional layer.
        padding_list (list[int] or list[tuple]): Padding values for each convolutional layer.
        in_channel (int): Number of input channels.
        p (float): Dropout rate applied after each convolution.
    """
    def __init__(self, num_filter_list, kernel_size_list, stride_list, padding_list, in_channel, p):
        super().__init__()
        self.num_conv = len(num_filter_list)
        if num_filter_list[-1] != in_channel:
            raise ValueError("Last element of num_filter_list must equal the number of input channels!")

        layers = []
        for i in range(self.num_conv):
            if i == 0:
                layers.append(
                    nn.Conv2d(
                        in_channel,
                        num_filter_list[0],
                        kernel_size=kernel_size_list[0],
                        stride=stride_list[0],
                        padding=padding_list[0]
                    )
                )
            else:
                layers.append(
                    nn.Conv2d(
                        num_filter_list[i - 1],
                        num_filter_list[i],
                        kernel_size=kernel_size_list[i],
                        stride=stride_list[i],
                        padding=padding_list[i]
                    )
                )
            layers.append(MCDropout(p))
            # Do not apply activation after the final convolution.
            if i != self.num_conv - 1:
                layers.append(nn.LeakyReLU())
        self.residual = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the residual block.

        Applies the residual block and adds the identity (input) to the block's output.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the residual block and adding the identity.
        """
        identity = x  # Identity mapping
        out = self.residual(x)
        out = out + identity
        return out


class UnrolledNetwork(nn.Module):
    """Neural network mapping measurements to the image manifold.

    This network iteratively refines a reconstruction estimate by applying a
    series of updates. Each update consists of a gradient step followed by a
    residual block replacing the proximal operator. The update at each iteration
    incorporates the measurement operator and its adjoint.

    Args:
        A (callable): The measurement operator. It should accept a tensor input.
        A_adjoint (callable): The adjoint of the measurement operator.
        lamb (float): Regularization parameter scaling the proximal gradient update.
        num_iter (int): Number of iterations (and residual blocks) to apply.
        residual_block_kwargs (dict): Keyword arguments to initialize the ResidualBlock.
            See the ResidualBlock documentation for details.
    """
    def __init__(self, A, A_adjoint, lamb, num_iter, residual_block_kwargs):
        super().__init__()
        self.A = A
        self.A_adjoint = A_adjoint
        self.lamb = nn.Parameter(torch.tensor([lamb], dtype=torch.float))
        self.num_iter = num_iter
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(**residual_block_kwargs) for _ in range(num_iter)]
        )

    def iteration(self, x, theta, i):
        """Perform one iteration of the proximal update.

        The update is computed by combining the current estimate with the
        measurement information, followed by applying the i-th residual block.

        Args:
            x (torch.Tensor): Measurement tensor.
            theta (torch.Tensor): Current reconstruction estimate.
            i (int): Iteration index to select the residual block.

        Returns:
            torch.Tensor: Updated reconstruction estimate.
        """
        gradient_update_result = theta - 2 * self.lamb * self.A_adjoint(self.A(theta)) + 2 * self.lamb * self.A_adjoint(x)
        out = self.residual_blocks[i](gradient_update_result)
        return out

    def forward(self, x, theta):
        """Forward pass of the unrolled network.

        Iteratively refines the reconstruction estimate using the measurement tensor.

        Args:
            x (torch.Tensor): Measurement tensor.
            theta (torch.Tensor): Initial reconstruction estimate.

        Returns:
            torch.Tensor: Final reconstruction estimate after all iterations.
        """
        out = self.iteration(x, theta, 0)
        for i in range(1, self.num_iter):
            out = self.iteration(x, out, i)
        return out


# Test the implementation of the unrolled network
if __name__ == '__main__':
    from torchsummary import summary

    # Define dummy measurement operator functions.
    def A(x):
        """Dummy measurement operator: identity."""
        return x

    def A_adjoint(x):
        """Dummy adjoint operator: identity."""
        return x

    # Define an example residual block configuration.
    residual_block_kwargs = {
        "num_filter_list": [64, 64, 3],
        "kernel_size_list": [3, 3, 3],
        "stride_list": [1, 1, 1],
        "padding_list": [1, 1, 1],
        "in_channel": 3,
        "p": 0.1,
    }

    # Instantiate the unrolled network.
    num_iter = 3  # For example, use 3 iterations.
    lamb = 0.5
    model = UnrolledNetwork(A, A_adjoint, lamb, num_iter, residual_block_kwargs)

    # Print the model summary.
    # For example, assume measurement and initial estimate are 3x256x256.
    summary(model, input_size=[(3, 256, 256), (3, 256, 256)])

    # Create dummy inputs for measurement (x) and initial reconstruction (theta).
    x = torch.randn(1, 3, 256, 256)      # Measurement tensor.
    theta = torch.randn(1, 3, 256, 256)  # Initial reconstruction.

    # Pass the inputs through the network.
    output = model(x, theta)
    print("Output shape:", output.shape)

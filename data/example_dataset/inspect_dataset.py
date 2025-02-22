import numpy as np
import matplotlib.pyplot as plt


def display_sample(npz_file, sample_index=0):
    """
    Loads an NPZ file, selects one sample, and displays:
      - The ground truth image's magnitude and phase.
      - The inverse FFT (reconstruction) from the subsampled Fourier
        coefficients's magnitude and phase.
      - The subsampling mask used for all images.

    The NPZ file is expected to contain:
      - 'ground_truth': an array of shape (N, 2, 256, 256), where the first
        channel represents the real part and the second channel represents the
        imaginary part of the ground truth complex image.
      - 'measurement': an array of shape (N, 2, 256, 256) of the subsampled
        normalized Fourier coefficients, with the first channel as the real part
        and the second as the imaginary part.
      - 'mask': a boolean numpy array of shape (256,256) used to subsample
        Fourier coefficients.

    Args:
        npz_file (str): Path to the NPZ file.
        sample_index (int): Index of the sample to display.
    """
    # Load the data from the NPZ file
    data = np.load(npz_file)
    ground_truth = data['ground_truth']  # shape: (N, 2, 256, 256)
    measurement = data['measurement']    # shape: (N, 2, 256, 256)
    mask = data['mask']                  # shape: (256,256)

    # Select the specified sample
    gt_sample = ground_truth[sample_index]
    meas_sample = measurement[sample_index]

    # Convert ground truth to complex representation
    gt_complex = gt_sample[0] + 1j * gt_sample[1]
    gt_magnitude = np.abs(gt_complex)
    gt_phase = np.angle(gt_complex)

    # Convert measurement to complex representation
    meas_complex = meas_sample[0] + 1j * meas_sample[1]

    # Compute the inverse Fourier transform with orthonormal normalization
    rec_complex = np.fft.ifft2(meas_complex, norm='ortho')
    rec_magnitude = np.abs(rec_complex)
    rec_phase = np.angle(rec_complex)

    # Create a 2x3 grid for displaying the images
    fig, axs = plt.subplots(2, 3, figsize=(9, 6))

    axs[0, 0].imshow(gt_magnitude, cmap='gray')
    axs[0, 0].set_title("Ground Truth Magnitude")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(gt_phase, cmap='gray')
    axs[0, 1].set_title("Ground Truth Phase")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(mask, cmap='gray')
    axs[0, 2].set_title("Subsampling Mask")
    axs[0, 2].axis("off")

    axs[1, 0].imshow(rec_magnitude, cmap='gray')
    axs[1, 0].set_title("Reconstructed Magnitude")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(rec_phase, cmap='gray')
    axs[1, 1].set_title("Reconstructed Phase")
    axs[1, 1].axis("off")

    # Optionally, hide the last subplot if not used
    axs[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    npz_file = "dataset/train.npz"
    index = 5
    display_sample(npz_file, index)

import os
import glob
import numpy as np
from PIL import Image


def process_image(image_path, img_size, mask):
    """
    Process a single image: read, resize, convert to grayscale, add noise,
    and compute a subsampled normalized Fourier transform using a common mask.

    Args:
        image_path (str): Path to the image file.
        img_size (tuple): Desired image size (width, height) after resizing.
        mask (ndarray): A boolean numpy array of shape img_size used to
            subsample Fourier coefficients.

    Returns:
        tuple: (measurement, ground_truth)
            - ground_truth: numpy array of shape (2, img_size[1], img_size[0])
                representing the toy complex image, where the first channel is
                the grayscale image and the second channel is random noise.
            - measurement: numpy array of shape (2, img_size[1], img_size[0])
                representing the subsampled normalized
                Fourier transform of the toy complex image, with real and
                imaginary parts concatenated.
    """
    # Open image, convert to grayscale, and resize
    img = Image.open(image_path).convert("L")
    img = img.resize(img_size)

    # Convert image to numpy array (shape: (256,256)) and normalize to [0,1]
    gray = np.array(img, dtype=np.float32) / 255.0

    # Generate random noise image with the same size
    noise = np.random.uniform(0, 2 * np.pi, size=img_size)

    # Create the ground truth complex image by using the image as the magnitude
    # and the random noise as the phase
    ground_truth_real = gray * np.cos(noise)
    ground_truth_imag = gray * np.sin(noise)
    ground_truth = np.stack([ground_truth_real, ground_truth_imag], axis=0)  # shape: (2, 256, 256)

    # Form a complex image from the two channels (real=gray, imag=noise)
    complex_image = ground_truth[0] + 1j * ground_truth[1]

    # Compute the 2D Fourier transform with orthonormal normalization
    fft_image = np.fft.fft2(complex_image, norm='ortho')

    # Apply the common subsampling mask
    subsampled_fft = fft_image * mask

    # Stack real and imaginary parts to form the measurement
    measurement = np.stack([np.real(subsampled_fft), np.imag(subsampled_fft)], axis=0)  # shape: (2,256,256)

    return measurement, ground_truth


def process_folder(folder_path, output_filename, img_size, mask):
    """
    Process all JPEG images in a folder and save the corresponding measurement,
    ground truth data, and subsampling mask to an NPZ file.

    Args:
        folder_path (str): Path to the folder containing JPEG images.
        output_filename (str): Filename for the output NPZ file.
        img_size (tuple): Desired image size (width, height) after resizing.
        mask (ndarray): A boolean numpy array of shape img_size used to
        subsample Fourier coefficients.
    """
    # Gather all JPEG images (covers both .jpg and .jpeg)
    image_paths = glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.jpeg'))

    measurements = []
    ground_truths = []

    for img_path in image_paths:
        meas, gt = process_image(img_path, img_size, mask)
        measurements.append(meas)
        ground_truths.append(gt)

    # Convert lists to numpy arrays
    measurements = np.array(measurements)
    ground_truths = np.array(ground_truths)

    # Save the data as an NPZ file with keys "measurement", "ground_truth", and "mask"
    np.savez_compressed(output_filename, measurement=measurements, ground_truth=ground_truths, mask=mask)
    print(f"Saved {len(image_paths)} samples to {output_filename}")


def main():
    # Root folder where BSD500 is located, with subfolders 'train', 'val', and 'test'
    root_folder = "BSDS500"
    modes = ['train', 'val', 'test']
    img_size = (256, 256)

    # Create output folder "dataset" if it does not exist
    output_folder = "dataset"
    os.makedirs(output_folder, exist_ok=True)

    # Generate a common subsampling mask for all images using a subsampling factor (e.g., 50%)
    subsampling_factor = 0.8
    mask = np.random.rand(*img_size) < subsampling_factor

    for mode in modes:
        folder = os.path.join(root_folder, mode)
        output_file = os.path.join(output_folder, f"{mode}.npz")
        print(f"Processing {mode} data from {folder}...")
        process_folder(folder, output_file, img_size, mask)

if __name__ == "__main__":
    main()

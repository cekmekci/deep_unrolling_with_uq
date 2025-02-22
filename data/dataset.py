import torch
import numpy as np
from torch.utils.data import Dataset

class ReconstructionDataset(Dataset):
    """Dataset for reconstruction problems.

    This dataset loads measurements and corresponding ground truth data from an NPZ file.
    The NPZ file is expected to contain arrays stored under the keys 'measurement' and 'ground_truth'.
    """

    def __init__(self, npz_file):
        """Initializes the ReconstructionDataset by loading data from an NPZ file.

        Args:
            npz_file (str): Path to the NPZ file containing the data. The file must contain arrays with keys
                'measurement' and 'ground_truth'.

        Raises:
            AssertionError: If the number of measurement samples does not equal the number of ground truth samples.
        """
        data = np.load(npz_file)
        self.measurements = data['measurement']
        self.ground_truth = data['ground_truth']

        assert len(self.measurements) == len(self.ground_truth), (
            "Number of measurement samples and ground truth samples must be equal."
        )

    def __len__(self):
        """Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return len(self.measurements)

    def __getitem__(self, idx):
        """Retrieves the measurement and corresponding ground truth sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple (measurement, ground_truth) where each element is a numpy array.
        """
        measurement = self.measurements[idx]
        ground = self.ground_truth[idx]

         # Convert the numpy arrays to PyTorch tensors
        measurement = torch.from_numpy(measurement).float()
        ground = torch.from_numpy(ground).float()

        return measurement, ground

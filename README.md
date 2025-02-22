# Deep Unrolling with Uncertainty Quantification

A Bayesian neural network-based deep unrolling framework for solving imaging inverse problems.

Table of Contents
-----------------
- Overview
- Features
- Project Structure
- Installation
- Usage
  - Training
  - Testing
- Toy Dataset
- Customizing for Other Problems
- Future Work
- License

Overview
--------
This repository contains an implementation of a learning-based image reconstruction framework that is capable of quantifying epistemic and aleatoric uncertainties, based on deep unrolling and Bayesian neural networks.

Features
--------
**Modular Structure**: The project is organized into distinct modules for data handling, training, and testing.

**Dataset**: The script "data/dataset.py" implements a custom dataset class that returns (measurement, true image) pairs upon call. An example dataset is provided under the "data" directory for illustrative purposes. The script in "data/example_dataset" generates a toy dataset. The toy dataset contains complex ground truth images and their corresponding measurements. The complex images are constructed such that the magnitude images are sourced from the BSDS500 dataset, and the phase images are generated randomly between 0 and 2π. For each complex ground truth image, the corresponding measurement is computed as the subsampled Fourier transform (with 50% of coefficients observed) of the ground truth image.

**Training and Testing Scripts**: The script "train.py" trains the framework, and "test.py" evaluates the framework on test data and displays some visual results.

**Forward Operator and Adjoint**: The script that generates the data, i.e., "data/example_dataset/create_dataset.py" implicitly defines the forward operator on Lines 47-51. For training and test purposes, this forward operator and its adjoint are explicitly defined within the training and testing scripts.

Project Structure
-----------------
The project is structured as follows:

    .
    ├── data                    # Custom PyTorch dataset subclass
        ├── dataset.py          # Custom PyTorch dataset subclass
        ├── example_dataset     # Toy dataset creation script and sample data
    ├── train.py                # Script to train the model
    ├── test.py                 # Script to evaluate the model and display results
    └── README.md

  data/


Usage
-----
**Generate Toy Data**: To generate the example data used within this repo, run:
```
cd data/example_dataset/
python create_dataset.py
```
The data generation script generates 3 npz files containing the training, validation, and the test datasets. Each npz file is a dictionary with three keys: "measurement", "ground_truth", and "mask".



**Training**: To train the uncertainty quantifying deep unrolled network on the toy dataset, run:
```
python -m train
```
The training script loads the data using the dataset class provided in "data/dataset.py" and defines the forward operator (A) and its adjoint (A_adjoint) explicitly although it is already implicitly defined in the "create_dataset.py" script. Then, it trains the model using the provided configuration file "config.yaml" and saves the trained model to the directory "checkpoints".

**Testing**: To evaluate the trained model and visualize the results, run:
```
python -m test
```
The test script loads the trained model based on the checkpoint instance chosen in the configuration file and performs inference. It displays the reconstructed image along with the uncertainty maps. We will include the  quantitative metrics soon.


Customizing for Other Problems
------------------------------
If you wish to adapt this code for a different imaging inverse problem:

1. Dataset Customization:
   - Generate your own dataset.
   - Update the "dataset.py" script in the "data" folder to handle your data format.

2. Operator Modification:
   - Modify the forward operator (A) and its adjoint (A_adjoint) in both "train.py" and "test.py" to suit your specific problem.

3. Architecture Modification:
   - Based on the nature of the inverse problem you are dealing with, you may consider modifying the neural network architectures provided in the models directory.

3. Configuration Modification:
   - Again, based on the nature of the inverse problem you are dealing with, you may consider modifying the training and inference configurations in the configuration file "config.yaml".

Future Work
-----------
- Quantitative Metrics: Integration of quantitative evaluation metrics will be added soon.

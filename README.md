# DA422 Presentation

This repository contains implementations and research related to neural network optimization techniques, focusing on BinaryConnect and Binary Neural Networks (BNNs). The project explores methods to enhance computational efficiency and reduce memory usage in deep learning models, with applications to datasets like MNIST, CIFAR-10, and SVHN.

## Directory Structure

- **Binary Connect/**
  - `README.md`: Overview and instructions for the BinaryConnect implementation.
  - `batch_norm.py`: Implementation of batch normalization layers.
  - `binary_connect.py`: Core BinaryConnect logic for training with binary weights.
  - `cifar10.py`: Script to train a CNN on the CIFAR-10 dataset using BinaryConnect.
  - `mnist.py`: Script to train an MLP on the MNIST dataset using BinaryConnect.
  - `svhn.py`: Script to train a CNN on the SVHN dataset using BinaryConnect.
  - `svhn_preprocessing.py`: Preprocessing script for the SVHN dataset.

- **Roleplay - Academic Researcher/**
  - `README.md`: Details on the Binary Neural Network implementation.
  - `binary_deterministic_stochastic.py`: Implementation of BNNs with deterministic and stochastic binarization.
  - `load_mnist.py`: Utility script to load and preprocess the MNIST dataset.
  - `non_binary.py`: Implementation of a traditional multi-layer neural network for comparison.

## Motivations

The project stems from the research paper "BinaryConnect: Training Deep Neural Networks with binary weights during propagations" (Courbariaux et al.), which proposes using binary weights to simplify computations in neural networks. This work extends to BNNs, replacing both weights and activations with binary values (+1 or -1) to further optimize memory and training time, achieving near state-of-the-art results on benchmark datasets.

## Requirements

- **Python**: 2.7 or 3.6.5 (depending on the script)
- **Libraries**:
  - NumPy
  - SciPy
  - Theano (bleeding edge version)
  - Pylearn2
  - Lasagne
  - PyTables (for SVHN)
  - Matplotlib
  - Scikit-learn
  - TensorFlow
  - Keras
- **Hardware**: NVIDIA GPU recommended (e.g., GTX 680, Titan Black) for faster training; CPU option available with patience.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shrey01022004-da422-presentation.git
   cd shrey01022004-da422-presentation
   ```

2. Install dependencies:
   ```bash
   pip install numpy scipy theano pylearn2 lasagne pytables matplotlib scikit-learn tensorflow keras
   ```

3. Set environment variables (e.g., for SVHN dataset):
   ```bash
   export PYLEARN2_DATA_PATH=/path/to/data
   export SVHN_LOCAL_PATH=/path/to/SVHN
   ```

## Usage

### BinaryConnect Experiments

- **MNIST**:
  ```bash
  python Binary Connect/mnist.py
  ```
  Trains an MLP on MNIST, expected test error
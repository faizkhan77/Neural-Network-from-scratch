# Simple Artificial Neural Network from Scratch

## Overview

This repository contains a Simple Artificial Neural Network (ANN) implemented from scratch using Object-Oriented Programming (OOP) principles. The project includes three optimization functions: batch gradient descent, mini-batch gradient descent, and stochastic gradient descent. The network has been tested on various datasets for both binary and multi-class classification tasks, demonstrating promising results.

## Features

- **Batch Gradient Descent**: Computes the gradient using the entire dataset, ensuring smooth convergence towards the minimum.
- **Mini-Batch Gradient Descent**: Balances efficiency and stability by computing gradients on subsets of the dataset (mini-batches).
- **Stochastic Gradient Descent (SGD)**: Computes gradients on individual examples, providing faster updates and the ability to navigate through local minima.

## Visualization

The implementation includes a feature to plot a chart showing how the cost decreases as the number of epochs increases, providing a clear visualization of the model's learning progress.

## Datasets

The neural network has been tested on the following datasets:
- Titanic dataset
- Parkinson's disease dataset
- Insurance dataset
- Employee salaries dataset

## Results

The results from these tests have been better than expected, especially for a first attempt at building a neural network from scratch.

## Repository Structure

- `ANN_from_scratch.py`: The main implementation of the Artificial Neural Network.
- `notebook/`: Contains Jupyter Notebooks used for testing, training, predicting, and evaluating the neural network.
  - `Testing_the_ANN.ipynb`: Notebook on Testing the ANN on Parkison's Disease dataset.
- `datasets/`: Contains datasets you can try.
  - `insurance.csv`: Insurance dataset.
  - `preprocessed_parkinson.csv`: Parkinson's disease dataset.
  - `preprocessed_salaries.csv`: Employee salaries dataset.
  - `processed_titanic.csv`: Titanic dataset.
- `README.md`: This file.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Required Python libraries (listed in `requirements.txt`)

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/ANN_from_scratch.git
   ```
2. Navigate to the project directory:
   ```sh
   cd Neural-Network-from-scratch
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Usage

1. Import the necessary modules and classes from the `NeuralNetwork.py` file.
2. Load your dataset and preprocess it as required.
3. Initialize and configure the ANN.
4. Train the ANN using one of the provided optimization methods.
5. Evaluate the ANN on your test data.

Example usage can be found in the Jupyter Notebooks provided in the `notebooks/` directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Thanks to Codebasics and other resources for providing excellent learning materials and inspiration.

## Contact

For any questions or suggestions, please feel free to contact me at faizkhan.net7@gmail.com.


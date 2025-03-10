# Structural Causal Discovery Toolkit (SD2C)

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

SD2C (Static Dependency to Causality) is a toolkit for conducting experiments in causal discovery, offering a comprehensive framework for generating synthetic datasets with known causal structures and applying various causal discovery algorithms to recover these structures.

This repository provides tools for:
- Generating synthetic datasets with known causal relationships
- Extracting various matrix representations (adjacency, correlation, partial correlation)
- Converting between different data representations
- Training deep learning models to infer causal relationships

## Repository Structure

```
├── src/
│   ├── generation/       # Synthetic data generation utilities
│   │   ├── builder.py    # Main class for dataset generation
│   │   └── functions.py  # Causal relationship functions
│   ├── models/           # Time series causal models
│   └── utils/            # Utility functions for data handling
├── par2ad.py             # Partial correlation to adjacency matrix conversion
├── data_generation.ipynb # Notebook demonstrating data generation
└── pipeline.ipynb        # End-to-end causal discovery pipeline
```

## Key Components

### Builder Class

The `Builder` class in `src/generation/builder.py` is the core data generation component, allowing users to create synthetic datasets with known causal structures. It supports:

- Multiple functional relationships (linear, polynomial, sigmoid, nonlinear, interaction)
- Customizable dataset sizes and variable counts
- Exogenous and endogenous variables
- Extraction of various matrix representations

### Causal Function Types

SD2C supports several types of causal relationships:

1. **Linear**: Standard linear relationships between variables
2. **Polynomial**: Nonlinear polynomial relationships with customizable degrees
3. **Sigmoid**: Sigmoidal nonlinear relationships
4. **Nonlinear**: Custom nonlinear functions (sin, tanh, etc.)
5. **Interaction**: Multiplicative interactions between variables

### ParcorrToAdjacencyModel Class

The `ParcorrToAdjacencyModel` in `par2ad.py` is a neural network framework for learning mappings from partial correlation matrices to adjacency matrices, enabling data-driven causal discovery.

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sd2c.git
cd sd2c

# Create and activate a virtual environment
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### Generating Synthetic Data

```python
from src.generation.builder import Builder
import math

# Initialize a Builder instance
builder = Builder(
    observations=250,             # Number of observations per dataset
    n_variables_exo=4,            # Number of exogenous variables
    n_variables_endo=6,           # Number of endogenous variables
    functions_to_use=[
        "linear",
        "polynomial",
        "sigmoid",
        "nonlinear",
        "interaction",
    ],
    functions_kwargs={
        "polynomial": [[1, 2], [2, 3]],
        'nonlinear': [math.sin, math.tanh],
    },
    datasets_per_function=10,     # Number of datasets per function type
    noise_std=0.2,                # Standard deviation of noise
    seed=123                      # Random seed for reproducibility
)

# Generate datasets
builder.build()

# Retrieve matrices
adjacency_matrix = builder.get_adjacency_matrix('linear', 0)
correlation_matrix = builder.get_correlation_matrix('linear', 0)
parcorr_matrix = builder.get_parcorr_matrix('linear', 0)
```

#### Training a Causal Discovery Model

```python
from par2ad import ParcorrToAdjacencyModel, generate_training_data

# Generate training data
parcorr_matrices, adjacency_matrices = generate_training_data(builder, n_samples=200)

# Create and train the model
model = ParcorrToAdjacencyModel(input_shape=(10, 10, 1), output_shape=(10, 10))

# Prepare the data
X_train, X_val, y_train, y_val = model.prepare_data(parcorr_matrices, adjacency_matrices)

# Train the model
history = model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=16)

# Test on new data
test_parcorr = builder.get_parcorr_matrix("linear", 0)
test_adjacency = builder.get_adjacency_matrix("linear", 0)
model.plot_comparison(test_parcorr.values, test_adjacency.values)
```

## Example Notebooks

The repository includes Jupyter notebooks that demonstrate the use of the toolkit:

1. **data_generation.ipynb**: Demonstrates how to generate synthetic data with different causal structures
2. **pipeline.ipynb**: Shows an end-to-end pipeline from data generation to causal discovery

## Time Series Models

The repository also includes several time series causal models in `src/models.py`:

1. Model 1: `-0.4 * (3 - (Y_bar_t[N_j])**2) / (1 + (Y_bar_t[N_j])**2) + 0.6 * (3 - (Y_bar_t-1[N_j] - 0.5)**3) / (1 + (Y_bar_t-1[N_j] - 0.5)**4) + W_t+1[j]`
2. Model 2: `(0.4 - 2 * exp(-50 * Y_bar_t-1[N_j]**2)) * Y_bar_t-1[N_j] + (0.5 - 0.5 * exp(-50 * Y_bar_t-2[N_j]**2)) * Y_bar_t-2[N_j] + W_t+1[j]`
3. Model 3: `1.5 * sin(pi / 2 * Y_bar_t-1[N_j]) - sin(pi / 2 * Y_bar_t-2[N_j]) + W_t+1[j]`

And many more, each representing different causal relationships in time series data.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Acknowledgments

- Inspired by advances in causal discovery research
- Based on structural causal models and time series analysis techniques
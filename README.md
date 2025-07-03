# Federated Air Quality Analysis

This project analyzes air quality data from multiple sites in Santiago, Chile. It provides two approaches: a centralized analysis and a federated analysis.

## Project Structure

- `data_completa11/`: This directory contains the dataset, with a separate CSV file for each monitoring station.
- `santiago_multi_site.ipynb`: A Jupyter notebook for a centralized analysis of the air quality data.
- `santiago_multi_site_federated.ipynb`: A Jupyter notebook for a federated analysis of the air quality data.
- `utils.py`: A Python script containing utility functions used by both notebooks.

## Getting Started

### Prerequisites

- Python 3
- Jupyter Notebook or JupyterLab
- Required Python libraries (e.g., pandas, matplotlib, jax, scienceplots, etc.)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1.  **Centralized Analysis:** Open and run the `santiago_multi_site.ipynb` notebook.
2.  **Federated Analysis:** Open and run the `santiago_multi_site_federated.ipynb` notebook.

## Data

The dataset consists of air quality data from 11 monitoring stations in Santiago, Chile. Each station's data is in a separate CSV file in the `data_completa11` directory.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


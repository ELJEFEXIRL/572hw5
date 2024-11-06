# 572hw5

# Lipophilicity Model Training with Morgan Fingerprints

This repository contains a script for training a machine learning model to predict lipophilicity using Morgan fingerprints and an MLPRegressor. The project is based on HW 4 and meets the requirements specified for HW 5.

## Repository Structure

- `src/`
  - `train_model.py` - The main script for training the model.
- `environment.yml` - Conda environment file to recreate the required environment (`572hw5`).
- `LICENSE` - License file for the project.
- `results.txt` - Generated after running the script; contains test RMSE, conda environment name, and hyperparameters.
- `README.md` - This file, containing project information and usage instructions.

## Dataset

The Lipophilicity dataset used in this project should be placed in the root directory and named `Lipophilicity.csv`. Ensure it contains the following columns:

- `smiles` - SMILES representations of the molecules.
- `exp` - Experimental lipophilicity values (logD).

## Requirements

- Anaconda or Miniconda installed on your system.
- Conda environment as specified in `environment.yml`.

## Installation and Setup

1. **Clone the Repository**

   ```bash
    git clone https://github.com/ELJEFEXIRL/572hw5.git
    cd 572hw5


##Create and Activate the Conda Environment:

Use the provided environment file to set up the conda environment with all dependencies:
bash
Copy code

conda env create -f environment.yml
conda activate 572hw5
Run the Model Training Script:

The main script is located in the folder and takes two command-line arguments:
hidden_layer_sizes: Specifies the sizes of the hidden layers in the MLP model (e.g., 100 50).
max_iter: Specifies the maximum number of iterations for the model training.

Example usage:
bash
Copy code
python train_model.py --hidden_layer_sizes 100 50 --max_iter 600

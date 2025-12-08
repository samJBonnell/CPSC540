# Generic Imports
import numpy as np
from pathlib import Path

# Personal Definitions
from us_lib.data.reader import load_records
from us_lib.data.parsing import extract_attributes

def data_loader(path : str):
    # Define input and output data locations
    input_path = Path(f"{path}/input.jsonl")
    output_path = Path(f"{path}/output.jsonl")

    if not input_path.exists() or not output_path.exists():
        print(f"\nInput or output path does not exist\nExiting")
        return
    
    # Load data records
    records = load_records(input_path, output_path)
    records, eigenvalues = extract_attributes(records, attributes= ['eigenvalue'])
    eigenvalues = eigenvalues['eigenvalue']

    # Extract parameters
    parameters = []

    for rec in records:
        row = [
            rec.input.num_longitudinal,
            rec.input.t_panel,                    
            rec.input.t_longitudinal_web,       
            rec.input.t_longitudinal_flange,
            rec.input.h_longitudinal_web,
            rec.input.w_longitudinal_flange
        ]
        parameters.append(row)

    X = np.array(parameters, dtype=float)
    y = np.array(eigenvalues, dtype=float)

    return X, y

def create_cnn_matrix(X : np.ndarray = None, rows : int = None, cols : int = None):
    """
    create_cnn_matrix takes as input a set of input and output data and creates set of CNN-compatible np.ndarrays as its return values
    
    Parameters:
    -----------
    X : np.ndarray (default : None)
        2D array corresponding to the input space where each column is a features
    rows : int (default : None)
        Number of rows in the output data structure
    cols : int (default : None)
        Number of cols in the output data structure
    
    Returns:
    --------
    X_cnn : np.ndarray
        4D matrix of size (X.shape[1] x X.shape[0] x rows x cols) where each layer X_cnn[i, j, :, :] = np.ones_like(X_cnn[i,j,:,:]) * X[i, j]
    
    """

    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    template_convolution = np.ones(shape=(rows, cols), dtype=float)
    X_cnn = np.ndarray(shape=(X.shape[0], X.shape[1], rows, cols))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_cnn[i, j, :, :] = (template_convolution.copy()) * X[i, j]
    
    return X_cnn
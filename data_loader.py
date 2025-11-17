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
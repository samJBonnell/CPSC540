# Linear Buckling Prediction of Stiffened Panels - CPSC540 Term Assignment
## Configuring your Environment
To start using the code in this repository, take the following actions.
1. Create a `conda` environment as follows (replace 'env_name' with environment name):
    ```cmd
    conda create --name env_name python=3.12
    ```
2. Install the required packages from `conda` except for `us_lib` and `torch`
3. Install `torch`, `torch_geometric`, etc. from `pip`
4. Install `us_lib` by running the following command in your working directory:
    ```cmd
    pip install -e .
    ```
    This will capture the library definition defined in `pyproject.toml` and will install `us_lib` locally in the current directory
5. Create a data directory `test_data/` and save all provided data into its own subfolder such as `set_1/`.
## Running the Code
To run the code, initially at least, use the following command:
```cmd
python3 data_loader.py --path test_data/set_1
```
Replace `python3` with `python` depending on aliases on your system.
## Future Work:
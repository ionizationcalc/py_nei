# NEI Simulation Wrapper

The `py_nei` project provides a Python interface for managing, executing, and visualizing simulations using the `nei_post.exe` executable.

## Project Structure

```
pynei/
├── NEI_Simulation.ipynb                 # Entry point / Usage example
├── requirements.txt        # Python dependencies
├── src/
│   ├── __init__.py
│   └── nei_simulation.py   # Core class definition (NEISimulation)
└── README.md               # Documentation
```

## Features

1.  **Configuration**: Manage paths (executable, working directory) and simulation parameters.
2.  **Execution**: Wrapper around `subprocess` to run `nei_exp3.exe` safely.
3.  **Visualization**: Framework for parsing output files and plotting results (using `matplotlib` is recommended).

## Usage

### 1. Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

### 1.1. Building the `nei_post.exe` Executable

To build the `nei_post.exe` executable required for simulations, please refer to the instructions provided in the [time_dependent_fortran repository](https://github.com/ionizationcalc/time_dependent_fortran).

### 2. Running a Simulation

See `NEI_Simulation.ipynb` for a complete example.

```python
from src.nei_simulation import NEISimulation

# Initialize
sim = NEISimulation(
    exe_path="/path/to/nei_post.exe",
    working_dir="./runs/run1"
)

# Configure
sim.set_parameters({"param_A": 10, "param_B": 20})
sim.generate_input_file()

# Run
sim.run_experiment()

# Visualize
sim.visualize_results()
```

## Customization

*   **Input Format**: Modify `NEISimulation.generate_input_file` in `src/nei_simulation.py` to match the specific format required by your `.exe`.
*   **Output Parsing**: Modify `NEISimulation.load_results` to parse the specific output structure of your program.
*   **Plotting**: Implement the plotting logic in `NEISimulation.visualize_results` using `matplotlib` or `seaborn`.

## License

See [LICENSE](LICENSE).


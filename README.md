# LightMat

LightMat is an easy-to-use Python module designed to calculate the light-induced potential of hyperfine atoms in the light field of arbitrary laser beams. This tool can be used to quantitatively find the trap depth of certain laser setups, calculate the differential light shift of different hyperfine levels, and perform calculations of hyperfine dynamical scalar-, vector-, and tensor-polarizabilities. Additionally, it can compute the spatial density of an atomic gas (bosons, fermions, or a mixture) in an arbitrary external trapping potential.

## Features
- **Comprehensive Calculations**: Supports calculations for hyperfine dynamical polarizabilities, light potential for arbitrary laser setups and spatial density in arbitrary potentials
- **User-Friendly**: Intuitive and easy to use.
- **Physical Quantities with Units**: All physical quantities come with units attached, easily convertible using Astropy in the background.
- **Plotting**: Capabilities for interactive plotting of 3d data arrays such as the calculated potential, overview of beams in a setup, plotting of calculated densities in 1d and 2d, etc.

## Installation

You can set up LightMat by either cloning the GitHub repository and either setting up a conda environment with the corresponding yaml file or by installing all necessary packages via pip with the requirements.txt file.

### Clone the GitHub Repository
```bash
git clone https://github.com/yourusername/LightMat.git
cd LightMat
```

### Using Conda
To set up the conda environment, run:
```bash
conda env create -f environment.yml
conda activate lightmat-env
```

### Using Pip
To install the necessary packages via pip, run:
```bash
pip install -r requirements.txt
```

## Usage

Here's a basic example of how to use LightMat:

```python
import lightmat as lm

# Example code to calculate the trap depth
trap_depth = lm.calculate_trap_depth(laser_parameters)
print(f"Trap Depth: {trap_depth}")

# Example code to calculate the differential light shift
light_shift = lm.calculate_differential_light_shift(atom_parameters)
print(f"Differential Light Shift: {light_shift}")

# Example code to plot the potential
lm.plot_potential(laser_parameters, atom_parameters)
```

For detailed usage and examples, please refer to the documentation.

## Contributing

If you find any bugs or have suggestions for improvements, please open an issue on GitHub.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any questions or inquiries, please contact:
- **Email**: [l.p.bleiziffer@gmail.com](mailto:l.p.bleiziffer@gmail.com)

## Acknowledgments

We appreciate contributions and feedback from the community to make LightMat better.

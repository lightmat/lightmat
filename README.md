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
git clone https://github.com/lightmat/lightmat.git
cd LightMat
```

### Using Conda
To set up the conda environment, run:
```bash
conda env create -f environment.yml
conda activate lightmat
```
After this step, the lightmat environment can also be selected as python kernel in a jupyter notebook, for example in the tutorial notebooks.

### Using Pip
To install the necessary packages via pip, run:
```bash
pip install -r requirements.txt
```

## Usage

There can be found a tutorial jupyter notebook for the potential calculation of a single hyperfine atom in the light field of some lasers and another tutorial notebook for the density calculation of an atomic gas in an arbitrary external potential.

## Contributing

If you find any bugs or have suggestions for improvements, please open an issue on GitHub.

## Contact

For any questions or inquiries, please contact:
- **Email**: [l.p.bleiziffer@gmail.com](mailto:l.p.bleiziffer@gmail.com)

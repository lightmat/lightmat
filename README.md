# LightMat

LightMat is an easy-to-use Python module designed to calculate the light-induced potential of hyperfine atoms in the light field of arbitrary laser beams. This tool can be used to quantitatively find the trap depth of certain laser setups, calculate the differential light shift of different hyperfine levels, and perform calculations of hyperfine dynamical scalar-, vector-, and tensor-polarizabilities. Additionally, it can compute the spatial density of an atomic gas (bosons, fermions, or a mixture) in an arbitrary external trapping potential.

An overview of some of the classes with some of their attributes and methods:
![Bildschirmfoto von 2024-07-23 09-36-19](https://github.com/user-attachments/assets/61209c1d-553b-4309-beaf-e4a5c73ba7c2)

**Example:** The potential of a K40 atom in its hyperfine groundstate F=9/2, mF=-9/2 in the laser field of a bowtie xy-lattice, and a 1d z-lattice with lightsheet for additional z-confinement. The interactive plot enables viewing the potential in different planes in 2d or along different directions in 1d. In the tutorial_single_atom.ipynb notebook it is shown how to set up the laser beams and atom and calculate this potential.
![Bildschirmfoto von 2024-07-23 09-46-24](https://github.com/user-attachments/assets/5c77e3e5-d9f1-40bf-a980-6c1b8f58bf53)


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

## Future Plans

- **Resonant or near-detuned features**, e.g. atom trajectory under radiation pressure (for MOT or Zeeman slower calculations)​

- **More beams**, e.g. Laguerre beam, Gaussian ring beam, etc.​

- Possibility to add **static E- and B-fields** ​

- Functionality to add **light intensity noise sampled from a camera** to the potential​

- **Improved finite temperature density calculations** beyond Thomas-Fermi approximation, e.g. by implementing free energy functional minimization (Tao Shi, Eugene Demler, and J Ignacio Cirac, PRL 125(18):180602, 2020)

## Contact

For any questions or inquiries, please contact:
- **Email**: [l.p.bleiziffer@gmail.com](mailto:l.p.bleiziffer@gmail.com)

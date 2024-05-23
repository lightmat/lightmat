import numpy as np
from scipy.integrate import simpson
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.units as u
from astropy.constants import hbar, k_B
from astropy.units import Quantity
from typing import Callable, Union, Sequence

from .bose_gas import BoseGas
from .fermi_gas import FermiGas
from .particle_props import ParticleProps
from .spatial_basis import SpatialBasisSet, GridSpatialBasisSet


class BoseFermiGas:
    def __init__(
            self, 
            bose_props: ParticleProps, 
            fermi_props: ParticleProps,
            a_bf: Union[float, Quantity],
            spatial_basis_set: Union[SpatialBasisSet, str] = 'grid',
            init_with_zero_T: bool = True,
            zero_T_threshold: float = 0.01,
            **basis_set_kwargs,
    ):
        """Initialize BoseFermiGas class. 
        
           Args:
               bose_props: Instance of ParticleProps class with species='boson'.
               fermi_props: Instance of ParticleProps class with species='fermion'.
               a_bf: s-wave scattering length between bosons and fermions. Can be a float or a Quantity with units of length.
               spatial_basis_set: Instance of SpatialBasisSet class or string, right now only 'grid' is supported. Defaults to 'grid'.
               init_with_zero_T: If True, run a zero-temperature calculation to improve initial guess for `mu`.
               zero_T_threshold: If the temperature is below this threshold, we assume T=0. Defaults to 0.01nK.
               **basis_set_kwargs: Keyword arguments for the SpatialBasisSet class.
        """
        self.bose_props = bose_props
        self.fermi_props = fermi_props
        self.a_bf = a_bf
        self.init_with_zero_T = init_with_zero_T
        self.zero_T_threshold = zero_T_threshold
        self.spatial_basis_set = spatial_basis_set
        self.basis_set_kwargs = basis_set_kwargs
        self._check_and_process_input("init")

        # Initialize BoseGase and FermiGas objects
        self.bose_gas = BoseGas(bose_props, spatial_basis_set, init_with_zero_T, **basis_set_kwargs)
        self.fermi_gas = FermiGas(fermi_props, spatial_basis_set, init_with_zero_T, **basis_set_kwargs)



    def eval_density(
            self,
            bosegas_kwargs_dict: dict = {},
            fermigas_kwargs_dict: dict = {},
            max_iter: int = 100,
            print_convergence_info_at_this_iteration: int = 0,
            show_progress: bool = True,
        ):
        # Approximations
        self.bose_gas.use_TF = True if 'use_TF' not in bosegas_kwargs_dict.keys() else bosegas_kwargs_dict['use_TF']
        self.bose_gas.use_HF = True if 'use_HF' not in bosegas_kwargs_dict.keys() else bosegas_kwargs_dict['use_HF']
        self.fermi_gas.use_TF_or_LDA = True if 'use_TF_or_LDA' not in fermigas_kwargs_dict.keys() else fermigas_kwargs_dict['use_TF_or_LDA']

        # Set up convergence history list for the plot_convergence_history() method
        self.bose_gas.convergence_history_mu = [self.bose_gas.mu.value]
        self.bose_gas.convergence_history_N = [self.bose_gas.N_particles]
        self.fermi_gas.convergence_history_mu = [self.fermi_gas.mu.value]
        self.fermi_gas.convergence_history_N = [self.fermi_gas.N_particles]
            
        # Run iterative procedure to update the densities
        iterator = tqdm(range(max_iter)) if show_progress else range(max_iter)
        for iteration in iterator: 
            pass

            




    def _check_and_process_input(
            self,
            which_method: str,
    ):
        if which_method == "init":
            # Initialize particle properties
            if isinstance(self.bose_props, ParticleProps):
                if self.bose_props.species != 'boson':
                    raise ValueError('bose_props must be a ParticleProps object with species="boson"')
            else:
                raise TypeError('bose_props must be a ParticleProps object')
            
            if isinstance(self.fermi_props, ParticleProps):
                if self.fermi_props.species != 'fermion':
                    raise ValueError('fermi_props must be a ParticleProps object with species="fermion"')
            else:
                raise TypeError('fermi_props must be a ParticleProps object')
            
            if self.bose_props.domain != self.fermi_props.domain:
                raise ValueError('bose_props and fermi_props must have the same domain')

            # Initialize a_bf
            if isinstance(self.a_bf, Quantity) and self.a_bf.unit.is_equivalent(u.m):
                self.a_bf = self.a_bf.to(u.m)
            elif isinstance(self.a_bf, (float, int)):
                self.a_bf = self.a_bf * u.m
            else:
                raise TypeError('a_bf must be a Quantity or a float')


            # Initialize spatial_basis_set
            if isinstance(self.spatial_basis_set, SpatialBasisSet):
                if self.spatial_basis_set.domain != self.bose_props.domain:
                    self.spatial_basis_set.domain = self.bose_props.domain
                    print("WARNING: spatial_basis_set domain was set to bose_props.domain, which is the same as fermi_props.domain.")
            elif isinstance(self.spatial_basis_set, str):
                if self.spatial_basis_set == 'grid':
                    num_grid_points = self.basis_set_kwargs.get('num_grid_points', 101) 
                    potential_function = self.basis_set_kwargs.get('potential_function', self.bose_props.V_trap)
                    self.spatial_basis_set = GridSpatialBasisSet(
                        self.bose_props.domain, 
                        num_grid_points,
                        potential_function,
                        )
                else:
                    raise NotImplementedError("Only 'grid' is implemented so far.")
            else:
                raise TypeError("spatial_basis_set must be a SpatialBasisSet object or a string.")
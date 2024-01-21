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
            **basis_set_kwargs,
    ):
        self.bose_props = bose_props
        self.fermi_props = fermi_props
        self.a_bf = a_bf
        self.spatial_basis_set = spatial_basis_set
        self.basis_set_kwargs = basis_set_kwargs
        self._check_and_process_input("init")

        # Initialize BoseGase and FermiGas objects
        self.bose_gas = BoseGas(bose_props, spatial_basis_set, init_with_zero_T, **basis_set_kwargs)
        self.fermi_gas = FermiGas(fermi_props, spatial_basis_set, init_with_zero_T, **basis_set_kwargs)



    def eval_density(
            self,
            bosons_use_TF: bool = False,
            bosons_use_Popov: bool = False,
            fermions_use_TF: bool = False,
            fermions_use_LDA: bool = True,
            max_iter: int = 1000,
            mu_convergence_threshold: float = 1e-5,
            N_convergence_threshold: float = 1e-3,
            mu_change_rate: float = 0.01,
            mu_change_rate_adjustment: int = 5,
            num_q_values: int = 50,
            cutoff_factor: float = 10,
            print_convergence_info_at_this_iteration: int = 0,
            show_progress: bool = True,
        ):
        # Approximations
        self.bosons_use_TF = bosons_use_TF
        self.bosons_use_Popov = bosons_use_Popov
        self.fermions_use_TF = fermions_use_TF
        self.fermions_use_LDA = fermions_use_LDA

        # Set up convergence history list for the plot_convergence_history() method
        self.bose_gas.convergence_history_mu = [self.bose_gas.mu.value]
        self.bose_gas.convergence_history_N = [self.bose_gas.N_particles]
        self.fermi_gas.convergence_history_mu = [self.fermi_gas.mu.value]
        self.fermi_gas.convergence_history_N = [self.fermi_gas.N_particles]
            
        # Run iterative procedure to update the densities
        iterator = tqdm(range(max_iter)) if show_progress else range(max_iter)
        for iteration in iterator: 
            # Initialize boson n_ex_array with zeros in first iteration
            if iteration == 0:
                    self.bose_gas.n_ex_array = np.zeros(self.spatial_basis_set.num_basis_funcs) * 1/u.um**3

            



    def _update_boson_density(
            self,
            num_q_values: int,
            cutoff_factor: float,
            mu_change_rate: float,
    ):
        # Update boson condensed density n0
        if self.bosons_use_TF:
            self.bose_gas._update_n0_with_TF_approximation()
        else:
            self.bose_gas._update_n0_with_E_functional_minimization()
            
        # Update boson non-condensed density n_ex if T>0 (otherwise we have n_ex=0)
        if self.bose_props.T.value > 1e-3:
            self.bose_gas._update_n_ex(num_q_values, cutoff_factor) # This uses either the semiclassical HF or Popov approximation
                                                                    # depending on the self.bosons_use_Popov flag.
            
        # Update the boson total density n = n0 + nex
        self.bose_gas.n_array = self.bose_gas.n0_array + self.bose_gas.n_ex_array

        # Update the boson particle numbers
        self.bose_gas.N_particles = self.spatial_basis_set.integral(self.bose_gas.n_array)
        self.bose_gas.N_particles_condensed = self.spatial_basis_set.integral(self.bose_gas.n0_array)
        self.bose_gas.N_particles_thermal = self.spatial_basis_set.integral(self.bose_gas.n_ex_array)
        self.bose_gas.condensate_fraction = self.bose_gas.N_particles_condensed / self.bose_gas.N_particles

        # Do soft update of the chemical potential mu based on normalization condition int dV (n0 + nex) = N_particles.
        # This increases mu, if N_particles is too small w.r.t N_particles_target and decreases mu if N_particles is
        # too large w.r.t. N_particles_target.
        new_mu_direction = (self.bose_props.N_particles - self.bose_gas.N_particles) / self.bose_props.N_particles * u.nK 
        self.bose_gas.mu += mu_change_rate * new_mu_direction

        # Calculate convergence info
        delta_mu_value = np.abs(self.bose_gas.mu.value - self.bose_gas.convergence_history_mu[-1]) 
        self.bose_gas.convergence_history_mu.append(self.bose_gas.mu.value)
        self.bose_gas.convergence_history_N.append(self.bose_gas.N_particles)

            # Print convergence info every other iteration
            if print_convergence_info_at_this_iteration > 0:
                if iteration % print_convergence_info_at_this_iteration == 0:
                    print(f"Iteration {iteration}:")
                    print('BOSONS:')
                    print('N: ', self.bose_gas.N_particles)
                    print('N_condensed: ', self.bose_gas.N_particles_condensed)
                    print('N_thermal: ', self.bose_gas.N_particles_thermal)
                    print('mu: ', self.bose_gas.mu)
                    print('delta_mu: ', delta_mu_value)
                    print('new_mu_direction: ', new_mu_direction)
                    print('mu_change_rate: ', mu_change_rate)
                    print("\n")

            # Dynamically adjust `mu_change_rate` based on recent changes
            if iteration % mu_change_rate_adjustment == 0 and iteration > 4:
                # Check for oscillations in mu
                oscillating = False
                for i in range(1, 5): # Check if mu was oscillating over last 4 iterations
                    if  (self.bose_gas.convergence_history_mu[-i] - self.bose_gas.convergence_history_mu[-i-1]) * \
                        (self.bose_gas.convergence_history_mu[-i-1] - self.bose_gas.convergence_history_mu[-i-2]) < 0:
                        oscillating = True
                        break
                if oscillating: # If oscillating, decrease the change rate to stabilize
                    mu_change_rate *= 0.5 
                else: # If not oscillating, increase the change rate to speed up convergence
                    mu_change_rate *= 2 

            # Check convergence criterion
            if delta_mu_value < mu_convergence_threshold*np.abs(self.bose_gas.mu.value) and \
               np.abs(self.bose_props.N_particles-self.bose_gas.N_particles) < N_convergence_threshold*self.bose_props.N_particles:
                if show_progress:
                    print(f"Convergence reached after {iteration} iterations.")
                break



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
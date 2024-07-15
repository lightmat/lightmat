import numpy as np
from scipy.integrate import simpson
from scipy.optimize import minimize
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.units as u
from astropy.constants import hbar, k_B
from astropy.units import Quantity
from matplotlib.figure import Figure
from typing import Union

from .particle_props import ParticleProps
from .spatial_basis import SpatialBasisSet, GridSpatialBasisSet


# TODO: Include overlap of basis in functional energy minimization calculation! For grid it's fine like this but not in general.

class BoseGas:
    """
    Class for the calculation of the density of a Bose gas in a given arbitrary external trapping potential. Positions r are
    assumed to be in units of [um]. The density is in units of [1/um**3]. The temperature and energy is in units of [nK]. 
    Throughout this class the astrophy units package is used to keep track of the units.

    The density of a Bose Gas consists of a condensed part n0(r) and a non-condensed part n_ex(r). 

    The condensed part is calculated in two possible ways:
    Either using the Thomas-Fermi approximation, i.e. n0(r) = max((mu-V_eff(r))/g*2, 0) with V_eff(r) = V_trap(r) + 2*g*n_ex(r).
    Or using energy functional minimization, i.e. minimizing the energy functional E[n0] = Ekin[n0] + Epot[n0] w.r.t. n0(r).

    The non-condensed part is calculated semiclassically using either the Hartree-Fock or Popov single-particle energy, i.e.
    n_ex(r) = int d^3p (2*pi*hbar)**-3 * 1/(exp((eps_p(r)-mu)/k_B/T) - 1) with eps_p(r) either given by the HF or Popov mean 
    field approximation.

    The total density is then given by n(r) = n0(r) + n_ex(r).

    The density is calculated iteratively by first updating n0(r) and then n_ex(r) and then repeating this procedure until
    convergence is reached. During the iterative procedure, the chemical potential mu is updated to ensure that the particle
    number is fixed. This is done by a soft update of mu based on the normalization condition int dV (n0 + nex) = N_particles.

    The iterative procedure is repeated until the chemical potential is converged or the maximum number of iterations is reached.

    ----------------------------------------------------------------------------------------------------------------------------------

    Attributes:
        particle_props: Instance of ParticleProps class with species='boson'.
        spatial_basis_set: Instance of SpatialBasisSet class.
        V_trap_array: Array of the external trapping potential in units of [nK].
        mu: Chemical potential in units of [nK].
        n0_array: Condensed density in units of [1/um**3].
        n_ex_array: Non-condensed density in units of [1/um**3].
        n_array: Total density in units of [1/um**3].
        N_particles: Total particle number.
        N_particles_condensed: Condensed particle number.
        N_particles_thermal: Non-condensed particle number.
        condensate_fraction: Condensate fraction.
        convergence_history_mu: List of chemical potential values during the iterative procedure.
        convergence_history_N: List of particle number values during the iterative procedure.
        use_TF: If True, use the Thomas-Fermi approximation to update n0, 
                if False, the energy functional is minimized w.r.t. n0 using CG descent.
        use_HF: If True, use the semiclassical Hatree Fock approximation to update n_ex,
                if False, use the semiclassical Popov approximation to update n_ex.

    ----------------------------------------------------------------------------------------------------------------------------------

    Methods:
        eval_density(): Run the iterative procedure to update in turn the densities `n_0_array` and `n_ex_array`.
        plot_convergence_history(): Plot the convergence history of the iterative procedure.
        plot_density_1d(): Plot the density along x-, y- and z-direction, i.e. n(x,0,0), n(0,y,0) and n(0,0,z).
        plot_density_2d(): Plot the density in the xy-, xz- and yz-plane, i.e. n(x,y,0), n(x,0,z) and n(0,y,z).
        plot_all(): Plot all of the above plots.

    ----------------------------------------------------------------------------------------------------------------------------------

    Note:
        The class uses units from the `astropy.units` module for physical quantities.
        It is important to ensure that the input parameters are given in the correct units or are appropriately converted.
    """
    def __init__(
            self, 
            particle_props: ParticleProps,
            spatial_basis_set: Union[SpatialBasisSet, str] = 'grid',
            init_with_zero_T: bool = True,
            zero_T_threshold: float = 0.01,
            **basis_set_kwargs,
        ) -> None:
        """Initialize BoseGas class. 
        
           Args:
               particle_props: Instance of ParticleProps class with species='boson'.
               spatial_basis_set: Instance of SpatialBasisSet class or string, right now only 'grid' is supported. Defaults to 'grid'.
               init_with_zero_T: If True, run a zero-temperature calculation to improve initial guess for `mu`.
               zero_T_threshold: If the temperature is below this threshold, we assume T=0. Defaults to 0.01nK.
               **basis_set_kwargs: Keyword arguments for the SpatialBasisSet class.
        """
        self.particle_props = particle_props
        self.spatial_basis_set = spatial_basis_set
        self.zero_T_threshold = zero_T_threshold
        self.init_with_zero_T = init_with_zero_T
        self.basis_set_kwargs = basis_set_kwargs
        self._check_and_process_input("init")

        # Initialize the external trapping potential
        self.V_trap_array = self.spatial_basis_set.get_coeffs(self.particle_props.V_trap)
        if isinstance(self.V_trap_array, Quantity):
            if self.V_trap_array.unit.is_equivalent(u.nK):
                self.V_trap_array = self.V_trap_array.to(u.nK)
            else:
                raise ValueError("V_trap must be in units of nK.")
        else:
            self.V_trap_array *= u.nK
        
        # Initilize the chemical potential using the TF approximation. In principle this would give:
        # mu(r) = V(r) + g*(n0(r) + 2*n_ex(r))
        # But we don't want a position-dependent mu, so as an initial guess we take:
        self.mu = np.min(self.V_trap_array) + self.particle_props.g.value*(self.particle_props.N_particles**(1/3))*u.nK 

        # Initialize the densities and particle numbers, which get assigned meaningful values after running eval_density()
        self.n_ex_array = np.zeros(self.spatial_basis_set.num_basis_funcs) * 1/u.um**3
        self.n0_array = None
        self._update_n0_with_TF_approximation()
        self.n_array = self.n0_array + self.n_ex_array
        self.N_particles = self.spatial_basis_set.integral(self.n_array)
        self.N_particles_condensed = self.spatial_basis_set.integral(self.n0_array)
        self.N_particles_thermal = self.spatial_basis_set.integral(self.n_ex_array)
        self.condensate_fraction = self.N_particles_condensed / self.N_particles

        # Run a zero-temperature calculation to improve initial guess for `mu`. If the provided temperature is 
        # already zero, we don't run this in order to not disturb the workflow of always running eval_density() 
        # after initializing this class.
        if self.init_with_zero_T and self.particle_props.T.value > self.zero_T_threshold:
            print("Perform zero temperature calculation with TF and HF approximaiton for initialization.")
            T = self.particle_props.T
            self.particle_props.T = 0 * u.nK 
            self.eval_density(use_TF=True, use_HF=True, show_progress=False)       
            self.particle_props.T = T
        

    def eval_density(
            self,
            use_TF: bool = True,
            use_HF: bool = True,
            init_with_TF: bool = True,
            max_iter: int = 100,
            mu_convergence_threshold: float = 1e-5,
            N_convergence_threshold: float = 1e-3,
            mu_change_rate: Union[float, None] = None,
            mu_change_rate_adjustment: int = 5,
            num_q_values: int = 50,
            cutoff_factor: float = 10,
            print_convergence_info_at_this_iteration: int = 0,
            show_progress: bool = True,
        ) -> None:
        """Run the iterative procedure to update in turn the densities `n_0_array` and `n_ex_array`. 
           The procedure is repeated until the chemical potential is converged or the maximum number 
           of iterations is reached. After running this method, the total density `n_array`, condensed 
           density `n0_array` and non-condensed density `n_ex_array` have meaningful values if the 
           convergence criterion was met. You can run `plot_convergence_history()` to see if you are 
           satisfied with the convergence.
        
           Args:
               use_TF: if True, use the Thomas-Fermi approximation to update n0, 
                       if False, the energy functional is minimized w.r.t. n0 using CG minimization. Defaults to True.
               use_HF: if True, use the semiclassical Hatree Fock approximation to update n_ex,
                       if False, use the semiclassical Popov approximation to update n_ex. Defaults to False. 
                       TODO: Fix division by zero problem in Popov implementation!
               max_iter: maximum number of iterations
               mu_convergence_threshold: We call the iterative procedure converged if the change in
                                         chemical potential from one iteration to the next is smaller than 
                                         `mu_convergence_threshold` times mu. Defaults to 1e-5.
               N_convergence_threshold: We call the iterative procedure converged if the change in particle number
                                        from one iteration to the next is smaller than `N_convergence_threshold` 
                                        times `N_particles_target`. Defaults to 1e-3.
               mu_change_rate: numerical hyperparameter for soft update of `mu`. Defaults to 0.01.
               mu_change_rate_adjustment: After so many iterations the `mu_change_rate` hyperparameter gets adjusted 
                                          for faster convergence. Defaults to 5.
               num_q_values: number of integrand evaluations for the simpson rule in momentum space integral to 
                             semiclassically calculate thermal density `n_ex_array`. Defaults to 50.
               cutoff_factor: factor to determine the cutoff momentum for the momentum space integral to semiclassically
                              calculate thermal density `n_ex_array`. The cutoff momentum is given by
                              p_cutoff = sqrt(cutoff_factor * 2*pi*m*k*T). Defaults to 10.
               print_convergence_info_at_this_iteration: if 0, don't print convergence info, if this value is i>0, 
                                                         then print convergence info every i-th iteration. Defaults to 0.
               show_progress: if True, use tqdm progress bar for iterative algorithm and print out the iteration after
                              which convergence was reached, if False, don't use progress bar and don't print out after 
                              which iteration convergence was reached. Defaults to True.
        """
        # Approximations
        self.use_TF = use_TF 
        self.use_HF = use_HF
        if self.use_TF:
            self.init_with_TF = False
        else:
            self.init_with_TF = init_with_TF

        # Set up convergence history list for the plot_convergence_history() method
        self.convergence_history_mu = [self.mu.value]
        self.convergence_history_N = [self.N_particles]

        # Store mu_change_rate (gets dynamically adjusted during iterative procedure to ensure convergence)
        if mu_change_rate is not None:
            self.mu_change_rate = mu_change_rate
        else: # Set default value if None is provided
            if not hasattr(self, 'mu_change_rate'):
                self.mu_change_rate = 0.01

        # Print some info if show_progress is True
        if show_progress:
            if self.init_with_TF and not self.use_TF:
                if self.particle_props.T.value == 0:
                    print("Initial n0 is calculated with the TF approximation. Afterwards functional energy minimization will be performed...")
                else:
                    raise NotImplementedError("RDMFT is not yet implemented. For T>0, please put use_TF=True.")
            elif self.use_TF:
                if self.use_HF:
                    print("Calculate n0 with TF approximation and n_ex with semiclassical HF approximation...")
                else:
                    print("Calculate n0 with TF approximation and n_ex with semiclassical Popov approximation...")
            else:
                if self.particle_props.T.value == 0:
                    if self.use_HF:
                        print("Calculate n0 with energy functional minimization and n_ex with semiclassical HF approximation...")
                    else:
                        print("Calculate n0 with energy functional minimization and n_ex with semiclassical Popov approximation...")
                else:
                    raise NotImplementedError("RDMFT is not yet implemented. For T>0, please put use_TF=True.")

        # Run iterative procedure to update the densities
        iterator = tqdm(range(max_iter)) if show_progress else range(max_iter)
        for iteration in iterator: 
            # Update the condensed and thermal density with the above specified approximations
            self._update_density_and_chem_potential(
                num_q_values, 
                cutoff_factor,
            )

            # Check convergence 
            converged = self._check_convergence(
                iteration, 
                print_convergence_info_at_this_iteration, 
                mu_convergence_threshold, 
                N_convergence_threshold, 
                show_progress,
            )
            if converged:
                break

            # Dynamically adjust `mu_change_rate` based on recent changes
            self._dynamically_adjust_mu_change_rate(
                iteration, 
                mu_change_rate_adjustment,
            )



    def _update_density_and_chem_potential(
            self,
            num_q_values,
            cutoff_factor,
        ) -> None:
        """Update the condensed and non-condensed density arrays `n0_array` and `n_ex_array` and the chemical potential `mu`.
           The update is done by first updating the condensed density `n0_array` and then the non-condensed density `n_ex_array`.
           The chemical potential `mu` is updated based on the normalization condition int dV (n0 + nex) = N_particles.
           
           Args:
                num_q_values: number of integrand evaluations for the simpson rule in momentum space integral to 
                              semiclassically calculate thermal density `n_ex_array`.
                cutoff_factor: factor to determine the cutoff momentum for the momentum space integral to semiclassically
                              calculate thermal density `n_ex_array`. The cutoff momentum is given by
                              p_cutoff = sqrt(cutoff_factor * 2*pi*m*k*T).
        """
        # Update condensed density n0
        if self.use_TF or self.init_with_TF:
            self._update_n0_with_TF_approximation()  
        else:
            if self.particle_props.T.value <= self.zero_T_threshold: # Energy functional minimization is only implemented for T=0
                self._calculate_n0_with_E_functional_minimization()
            else:
                raise NotImplementedError("Energy functional minimization is only implemented for T=0. RDMFT is not implemented yet.")
            
        # Update non-condensed density n_ex if T>0 (otherwise we have n_ex=0)
        if self.particle_props.T.value > self.zero_T_threshold:
            self._update_n_ex(num_q_values, cutoff_factor) # This uses either the semiclassical HF or Popov approximation
                                                           # depending on the self.use_HF flag.
            
        # Update the total density n = n0 + nex
        self.n_array = self.n0_array + self.n_ex_array

        # Update the particle numbers
        self.N_particles = self.spatial_basis_set.integral(self.n_array)
        self.N_particles_condensed = self.spatial_basis_set.integral(self.n0_array)
        self.N_particles_thermal = self.spatial_basis_set.integral(self.n_ex_array)
        self.condensate_fraction = self.N_particles_condensed / self.N_particles 

        # Do soft update of the chemical potential mu based on normalization condition int dV (n0 + nex) = N_particles.
        # This increases mu, if N_particles is too small w.r.t N_particles_target and decreases mu if N_particles is
        # too large w.r.t. N_particles_target.
        new_mu_direction = (self.particle_props.N_particles - self.N_particles) / self.particle_props.N_particles * u.nK 
        self.mu += self.mu_change_rate * new_mu_direction 
    


    def _check_convergence(
            self,
            iteration,
            print_convergence_info_at_this_iteration,
            mu_convergence_threshold,
            N_convergence_threshold,
            show_progress,
        ) -> None:
        """Check if the iterative procedure has converged. The convergence criterion is met if the change in chemical potential
           `mu` from one iteration to the next is smaller than `mu_convergence_threshold` times `mu` and the change in particle
           number `N_particles` from one iteration to the next is smaller than `N_convergence_threshold` times `N_particles`.
           
           Args:
                iteration: current iteration number.
                print_convergence_info_at_this_iteration: if 0, don't print convergence info, if this value is i>0, 
                                                          then print convergence info every i-th iteration.
                mu_convergence_threshold: We call the iterative procedure converged if the change in
                                          chemical potential from one iteration to the next is smaller than 
                                          `mu_convergence_threshold` times mu.
                N_convergence_threshold: We call the iterative procedure converged if the change in particle number
                                         from one iteration to the next is smaller than `N_convergence_threshold` 
                                         times `N_particles_target`.
                show_progress: if True, print out when convergence was reached.
        """
        # Calculate convergence info
        delta_mu_value = np.abs(self.mu.value - self.convergence_history_mu[-1]) 
        self.convergence_history_mu.append(self.mu.value)
        self.convergence_history_N.append(self.N_particles)

        # Print convergence info every other iteration
        if print_convergence_info_at_this_iteration > 0:
            if iteration % print_convergence_info_at_this_iteration == 0:
                print(f"Iteration {iteration}:")
                print('N: ', self.N_particles)
                print('N_condensed: ', self.N_particles_condensed)
                print('N_thermal: ', self.N_particles_thermal)
                print('mu: ', self.mu)
                print('mu_change_rate: ', self.mu_change_rate)
                print("\n")


        # Check convergence criterion
        if delta_mu_value < mu_convergence_threshold*np.abs(self.mu.value) and \
           np.abs(self.particle_props.N_particles-self.N_particles) < N_convergence_threshold*self.particle_props.N_particles:
            if self.init_with_TF and not self.use_TF:
                self.init_with_TF = False # set to False to continue with energy functional minimization
            if show_progress:
                print(f"Convergence reached after {iteration} iterations.")
            return True
        
        return False




    def _dynamically_adjust_mu_change_rate(
            self,
            iteration,
            mu_change_rate_adjustment,
        ) -> None:
        """Dynamically adjust the `mu_change_rate` hyperparameter based on recent changes in the chemical potential `mu`.
           If the chemical potential `mu` was oscillating over the last 4 iterations, decrease the change rate to stabilize,
           otherwise increase the change rate to speed up convergence.
           
           Args:
                iteration: current iteration number.
                mu_change_rate_adjustment: After so many iterations the `mu_change_rate` hyperparameter gets adjusted .
        """
        if iteration % mu_change_rate_adjustment == 0 and iteration > 4:
            # Check for oscillations in mu
            oscillating = False
            for i in range(1, 5): # Check if mu was oscillating over last 4 iterations
                if  (self.convergence_history_mu[-i] - self.convergence_history_mu[-i-1]) * \
                    (self.convergence_history_mu[-i-1] - self.convergence_history_mu[-i-2]) < 0:
                    oscillating = True
                    break
            if oscillating: # If oscillating, decrease the change rate to stabilize
                self.mu_change_rate *= 0.5 
            else: # If not oscillating, increase the change rate to speed up convergence
                self.mu_change_rate *= 2 


    def _update_n0_with_TF_approximation(
            self,
            V_additional_array: Union[Quantity, None] = None
        ) -> None:
        """Apply the Thomas-Fermi approximation to update the condensed density `n_0_array`, i.e.
           n0(r) = max((mu-V_eff(r))/g*2, 0) with V_eff(r) = V_trap(r) + 2*g*n_ex(r).
           If there is an additional potential V_additional(r), it gets added to V_eff(r) as well.

           Args:
                V_additional_array: additional potential to be added to the effective potential. Defaults to None.
        """
        # Add additional potential to V_eff if provided
        V_eff_array = self.V_trap_array + 2 * self.particle_props.g * self.n_ex_array
        if V_additional_array is not None:
            V_eff_array = V_eff_array + V_additional_array

        # Calculate n0(r) with TF approximation
        self.n0_array = np.maximum((self.mu - V_eff_array) / self.particle_props.g, 0)



    def _calculate_n0_with_E_functional_minimization(
            self,
        ) -> None:
        """Minimize the energy functional E[n0] = Ekin[n0] + Epot[n0] w.r.t. n0(r) to update the
           condensed density `n_0_array` with

           Ekin[n0] = int d^3r 1/8 * (hbar**2/(2*m) * (grad(n0(r)))**2 / n0(r) / k_B) 
           Epot[n0] = int d^3r V_eff(r) * n(r) with V_eff(r) = (V_trap(r) + g*(n0(r) + n_ex(r))

          The minimization is done using scipy.optimize.minimize().
        """
        assert self.n0_array.unit == 1/u.um**3, "n_array must be in units of 1/um**3."

        # Calculate n0(r) by minimizing the energy functional E[n0] = Ekin[n0] + Epot[n0]
        bounds = [(0, None) for _ in range(self.spatial_basis_set.num_basis_funcs)] # n0(r) must be positive everywhere
        E_min = minimize(self._E_functional, self.n0_array.value, jac=self._E_functional_deriv, bounds=bounds, method='L-BFGS-B')
        self.n0_array = E_min.x * 1/u.um**3

        # Normalize n(r) to the total particle number N_particles, this only works at T=0. At T>0 we need something
        # more sophisticated depending on the chemical potential mu, e.g. RDMFT.
        self.n0_array = self.n0_array * self.particle_props.N_particles / self.spatial_basis_set.integral(self.n0_array)

        

    def _E_functional(
            self,
            n0_array: np.ndarray,
            integrand_additional_array: Union[Quantity, None] = None,
    )-> Union[Quantity, float]:
        """Calculate the energy functional E[n0] = Ekin[n0] + Epot[n0] for a given condensed density `n0_array` with

           Ekin[n0] = int d^3r 1/8 * (hbar**2/(2*m) * (grad(n0(r)))**2 / n0(r) / k_B) 
           Epot[n0] = int d^3r V_eff(r) * n(r) with V_eff(r) = (V_trap(r) + g*(n0(r) + n_ex(r))

           It is possible to add an additional integrand of the form V(r)*n(r) for some potential and some density to the Epot[n0] 
           term. This can be useful to include e.g. interactions with a density of fermions. 

           Args:
                n0_array: condensed density n0(r) in units of [1/um**3]
                integrand_additional_array: additional integrand to be added to the Epot[n0] term. Defaults to None.

           Returns:
                E: Energy functional E[n0] in units of [nK]
        """
        # Convert n0(r) to units of [1/um**3] if it has no unit attached
        unitless = False
        if not isinstance(n0_array, Quantity):
            n0_array = n0_array * 1/u.um**3
            unitless = True

        # Calculate the kinetic energy term in the E functional
        # I assume that n0(r) defines a boundary surface in 3d space where it becomes zero and that (grad(n0(r)))**2 / n0(r)
        # is zero beyond that surface. Thus, we can integrate only over the region where n0(r) is non-zero, i.e. mask out the 
        # region where n0(r) is zero.
        integrand_array = np.zeros(self.spatial_basis_set.num_basis_funcs) * 1/u.um**5
        mask = n0_array > 0
        integrand_array[mask] = self.spatial_basis_set.gradient_dot_gradient(n0_array)[mask] / n0_array[mask] 
        Ekin = 1/8 * ((hbar**2/(2*self.particle_props.m) / k_B) * self.spatial_basis_set.integral(integrand_array)).to(u.nK)

        # Calculate the potential energy term in the E functional
        integrand_array = self.V_trap_array * (n0_array + self.n_ex_array) + \
                          1/2 * self.particle_props.g * self.spatial_basis_set.power((n0_array + self.n_ex_array), 2)
        if integrand_additional_array is not None:
            integrand_array = integrand_array + integrand_additional_array
        Epot = self.spatial_basis_set.integral(integrand_array).to(u.nK)

        # Return total energy, if no unit was attached to n0_array return energy also without unit
        if unitless:
            return (Ekin + Epot).value
        else:
            return Ekin + Epot

        

    def _E_functional_deriv(
            self,
            n0_array: np.ndarray,
            V_additional_array: Union[Quantity, None] = None,
    ) -> Union[Quantity, float]:
        """Calculate the functional derivative of the energy functional E[n0] = Ekin[n0] + Epot[n0] w.r.t. n0(r), i.e.
           dE/dn0(r) = dEkin/dn0(r) + dEpot/dn0(r) with
           
           dEkin/dn0(r) = -1/8 * (hbar**2/(2*m) * grad(n0(r)))**2 / n0(r)**2 
           dEpot/dn0(r) = V_eff(r) with V_eff(r) = V_trap(r) + g*n0(r)

           It is possible to add an additional potential V(r) to the Epot[n0] term. This can be useful to include e.g. 
           interactions with a density of fermions. 

           Note, that the functional derivative of the energy functional evaluated at the true density n0(r) corresponds 
           to the local chemical potential mu(r)! If we evaluate the functional derivative at the true density
           n0(r) at the position r corresponding to the minimum of the effective potential V_eff(r), we obtain the global
           chemical potential mu, i.e. the energy necessary to add one particle to the system keeping entropy and volume 
           fixed.

           Args:
                n0_array: condensed density n0(r) in units of [1/um**3]
                V_additional_array: additional potential to be added to the effective potential. Defaults to None.    

           Returns:       
                deriv_E: functional derivative of the energy functional E[n0] w.r.t. n0(r) in units of [nK]
        """
        unitless = False
        if not isinstance(n0_array, Quantity):
            n0_array = n0_array * 1/u.um**3
            unitless = True

        # Calculate functional derivative of the kinetic energy term in the E functional w.r.t. n0(r)
        # I assume that n0(r) defines a boundary surface in 3d space where it becomes zero and that (grad(n0(r)))**2 / n0(r)**2
        # as well as laplacian(n0(r)) / n0(r) are zero beyond that surface. Thus, we can integrate only over the region where 
        # n0(r) is non-zero, i.e. mask out the region where n0(r) is zero.
        deriv_Ekin = np.zeros(self.spatial_basis_set.num_basis_funcs) * 1/u.um**2
        mask = n0_array > 0
        deriv_Ekin[mask] = self.spatial_basis_set.gradient_dot_gradient(n0_array)[mask] / self.spatial_basis_set.power(n0_array,2)[mask] \
                          + 2 * self.spatial_basis_set.laplacian(n0_array)[mask] / n0_array[mask]
        deriv_Ekin = -1/8 * (hbar**2/(2*self.particle_props.m) * deriv_Ekin / k_B).to(u.nK)

        # Calculate functional derivative of the potential energy term in the E functional w.r.t. n0(r)
        deriv_Epot = (self.V_trap_array + self.particle_props.g * n0_array).to(u.nK)
        if V_additional_array is not None:
            deriv_Epot = deriv_Epot + V_additional_array

        # Return functional derivative of the total energy w.r.t. n0(r). If no unit was attached to n0_array, return also no unit
        if unitless:
            return (deriv_Ekin + deriv_Epot).value
        else:
            return deriv_Ekin + deriv_Epot



    def _update_n_ex(
            self,
            num_q_values: int = 50,
            cutoff_factor: float = 10,
            V_additional_array: Union[Quantity, None] = None,
        ) -> None:
        """Apply the semiclassical Hartree-Fock or Popov approximation to update the non-condensed 
           density `n_ex_array`, i.e.

           n_ex(r) = int d^3p (2*pi*hbar)**-3 * 1/(exp((eps_p(r)-mu)/k_B/T) - 1)

           where the single-particle energy eps_p(r) is given by either the semiclassical HF or Popov approximation.
           
           Args:
                num_q_values: number of integrand evaluations for the simpson rule to integrate over momentum space. 
                              Defaults to 50.
                cutoff_factor: factor to determine the cutoff momentum for the momentum space integral to semiclassically
                              calculate thermal density `n_ex_array`. The cutoff momentum is given by
                              p_cutoff = sqrt(cutoff_factor * 2*pi*m*k*T). Defaults to 10.
                V_additional_array: additional potential to be added to the effective potential. Defaults to None.
        """
        # Since we are at low temperature, we don't want to integrate over the entire momentum space, but only over low p 
        # as the BE distribution goes to zero very quickly for large p. Thus we set a cutoff at p_cutoff = sqrt(2*pi*m*k*T) 
        # which corresponds to the momentum scale of the thermal de Broglie wavelength (i.e. lambda_dB = h/p_cutoff).
        p_cutoff = np.sqrt(cutoff_factor * 2*np.pi*self.particle_props.m*k_B*self.particle_props.T).to(u.u*u.m/u.s) 
        # use this momentum units to deal with numerical numbers roughly on the order ~1

        # Also, for increased numerical stability, we can rescale the integral to the interval 
        # [0,1] and integrate over the dimensionless variable q = p/p_cutoff instead of p.
        q_values = np.linspace(0, 1, num_q_values) 
        q_values = q_values[:, np.newaxis] # reshape to broadcast with spatial basis set later

        # Calculate integrand values for each spatial basis function and each q value
        if self.use_HF:
            integrand_values = self._integrand_HF(q_values, p_cutoff, V_additional_array)
        else:
            integrand_values = self._integrand_Popov(q_values, p_cutoff, V_additional_array)

        # Check if integrand is zero at q=1 to ensure the cutoff momentum is large enough
        max_integrand_val = np.max(integrand_values)
        max_integrand_val_at_q1 = np.max(integrand_values[-1,:])
        if max_integrand_val_at_q1 > 1e-10 * max_integrand_val: # Check if integrand is not zero at q=1
            print("WARNING: Integrating until q=1, but integrand is not zero at q=1. Consider increasing cutoff_factor.")

        # Integrate using Simpson's rule to exploit vectorized implementation, which is not possible with scipy.integrate.quad()
        integral = simpson(integrand_values, q_values.flatten(), axis=0)

        # Update n_ex_array 
        self.n_ex_array = np.maximum((integral*(p_cutoff.unit)**3 / (2*np.pi * hbar)**3).to(1/u.um**3), 0)



    def _integrand_HF(
            self, 
            q: Union[float, np.ndarray],
            p_cutoff: Quantity,
            V_additional_array: Union[Quantity, None] = None,
        ) -> np.ndarray:
        """Calculate the integrand for the semiclassical momentum space integration with the HF single-particle energy, i.e.
         
           integrad = 1/(exp((eps_p(r)-mu)/k_B*T) - 1) with eps_p(r) = p^2/(2m) + V_eff(r) and 
           V_eff(r) = V_trap(r) + 2*g*(n0(r) + n_ex(r))
            
           It is possible to add an additional potential V(r) to the V_eff(r) term. 
           In principle, we have a 3d volume-integral in momentum space, but luckily it only depends on the 
           magnitude of the momentum, so we can integrate over the angles analytically giving us the required 
           integrand as 4*pi*p^2*f(eps_p-mu). Since we rescale the integral to integrate over the dimensionless 
           variable q = p/p_cutoff, we additionally need to multiply the integrand by p_cutoff.
        
           Args:
                q: dimensionless integration variable defined as q = p/p_cutoff
                p_cutoff: cutoff momentum for the integration
                
           Returns:
                f: Integrand p_cutoff*4*pi*p^2*f(eps_p) for the 1d integration over q in the interval [0,1]
        """
        # Momentum p is given by the dimensionless integration variable q times the cutoff momentum
        p = q * p_cutoff

        # Calculation of eps_p(r) in semiclassical HF approximation (eq. 8.115). 
        V_eff_array = self.V_trap_array + 2*self.particle_props.g * (self.n0_array + self.n_ex_array)
        if V_additional_array is not None:
            V_eff_array = V_eff_array + V_additional_array
        eps_p = (p**2/(2*self.particle_props.m) / k_B).to(u.nK) + V_eff_array

        # Scaled integrand for the 1d integral over q in the interval [0,1]
        return p_cutoff.value * 4*np.pi*p.value**2 / (np.exp((eps_p-self.mu) / self.particle_props.T) - 1) 



    def _integrand_Popov(
            self, 
            q: Union[float, np.ndarray],
            p_cutoff: Quantity,
            V_additional_array: Union[Quantity, None] = None,
        ) -> np.ndarray:
        #TODO: Somehow make sure that there is no divion by zero... (happens if chemical potential is too high
        #      such that eps_p becomes zero. 
        """Calculate the integrand for the semiclassical momentum space integration with the Popov single-particle energy, i.e.

           integrad = 1/(exp((eps_p(r)-mu)/k_B*T) - 1) with eps_p(r) = sqrt((p^2/(2m) + V_eff(r) - mu)^2 - (g*n0(r))^2) and
           V_eff(r) = V_trap(r) + 2*g*(n0(r) + n_ex(r))

           It is possible to add an additional potential V(r) to the V_eff(r) term.
           In principle, we have a 3d volume-integral in momentum space, but luckily it only depends on the
           magnitude of the momentum, so we can integrate over the angles analytically giving us the required
           integrand as 4*pi*p^2*f(eps_p-mu). Since we rescale the integral to integrate over the dimensionless
           variable q = p/p_cutoff, we additionally need to multiply the integrand by p_cutoff.

           Args:
                q: dimensionless integration variable defined as q = p/p_cutoff
                p_cutoff: cutoff momentum for the integration

           Returns:
                f: Integrand p_cutoff*4*pi*p^2*f(eps_p) for the 1d integration over q in the interval [0,1]
        """
        # Momentum p is given by the dimensionless integration variable q times the cutoff momentum
        p = q * p_cutoff

        # Calculation of eps_p(r) in semiclassical Popov approximation (eq. 8.119 in Pethick & Smith). 
        V_eff_array = self.V_trap_array + 2*self.particle_props.g * (self.n0_array + self.n_ex_array)
        if V_additional_array is not None:
            V_eff_array = V_eff_array + V_additional_array
        eps_p = np.sqrt(np.maximum(((p**2/(2*self.particle_props.m) / k_B).to(u.nK) + V_eff_array - self.mu)**2 - \
                        (self.particle_props.g*self.n0_array)**2, 0))

        num_non_condensed = ((p**2/(2*self.particle_props.m) / k_B).to(u.nK) + self.V_trap_array + \
                              2*self.particle_props.g*(self.n0_array + self.n_ex_array) - self.mu) / eps_p
        
        return p_cutoff.value*4*np.pi*p.value**2 * num_non_condensed / (np.exp(eps_p / self.particle_props.T) - 1)



    def plot_convergence_history(
            self,
            start: int = 0,
            end: int = -1,
            **kwargs,
        ) -> Figure:
        """Plot the convergence history of the iterative procedure carried out in `eval_density()`.

           Args:
                start: start index of the convergence history list to be plotted. Defaults to 0.
                end: end index of the convergence history list to be plotted. Defaults to -1, i.e. last index.
                **kwargs: keyword arguments for the plot, supported are 'title' and 'filename'.

           Return:
                fig: matplotlib figure object.
        """
        if (self.particle_props.T.value < self.zero_T_threshold and self.N_particles is not None) or np.linalg.norm(self.n_ex_array) > 0:
            title = kwargs.get('title', 'Convergence history, T='+str(self.particle_props.T)+', N='+str(int(self.N_particles)))  
            filename = kwargs.get('filename', None)

            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(title, fontsize=24)
            
            axs[0].plot(self.convergence_history_mu[start:end], c='k', marker='o', label=r'$\mu$')
            axs[0].set_xlabel('Iteration i', fontsize=14)
            axs[0].set_ylabel(r'$\mu_i \; \left[ k_B \times nK \right]$', fontsize=14)
            axs[0].set_title(r'Chemical potential $\mu$', fontsize=18)
            axs[0].grid(True)

            axs[1].plot(self.convergence_history_N[start:end], c='k', marker='o', label=r'$N$')
            axs[1].set_xlabel('Iteration i', fontsize=14)
            axs[1].set_ylabel(r'$N_i$', fontsize=14)
            axs[1].set_title(r'Total particle number $N$', fontsize=18)
            axs[1].grid(True)

            fig.tight_layout()
            if filename != None:
                fig.savefig(filename, dpi=300, bbox_inches='tight')

            return fig
        
        else:
            print("No convergence history found. Please run eval_density() first.")


    def plot_density_1d(
            self, 
            num_points: int = 200,
            **kwargs,
        ) -> Figure:
        """Plot the spatial density n(x,0,0), n(0,y,0) and n(0,0,z) along each direction respectively.
        
           Args:
                num_points: number of points along each direction to evaluate the density at. Defaults to 200.
                **kwargs: keyword arguments for the plot, supported are 'title' and 'filename'.

           Returns:
                fig: matplotlib figure object.
        """ 
        if (self.particle_props.T.value < self.zero_T_threshold and self.N_particles is not None) or np.linalg.norm(self.n_ex_array) > 0:
            title = kwargs.get('title', 'Spacial density, T='+str(self.particle_props.T)+', N='+str(int(self.N_particles)))  
            filename = kwargs.get('filename', None) 

            fig, axs = plt.subplots(1, 3, figsize=(14, 6))
            plt.subplots_adjust(top=0.85)
            fig.suptitle(title, fontsize=24)

            x = np.linspace(self.particle_props.domain[0][0].value, self.particle_props.domain[0][1].value, num_points)
            y = np.linspace(self.particle_props.domain[1][0].value, self.particle_props.domain[1][1].value, num_points)
            z = np.linspace(self.particle_props.domain[2][0].value, self.particle_props.domain[2][1].value, num_points)

            nx = self._check_n(self.spatial_basis_set.expand_coeffs(self.n_array, x, 0, 0))
            n0x = self._check_n(self.spatial_basis_set.expand_coeffs(self.n0_array, x, 0, 0))
            n_exx = self._check_n(self.spatial_basis_set.expand_coeffs(self.n_ex_array, x, 0, 0))

            ny = self._check_n(self.spatial_basis_set.expand_coeffs(self.n_array, 0, y, 0))
            n0y = self._check_n(self.spatial_basis_set.expand_coeffs(self.n0_array, 0, y, 0))
            n_exy = self._check_n(self.spatial_basis_set.expand_coeffs(self.n_ex_array, 0, y, 0))

            nz = self._check_n(self.spatial_basis_set.expand_coeffs(self.n_array, 0, 0, z))
            n0z = self._check_n(self.spatial_basis_set.expand_coeffs(self.n0_array, 0, 0, z))
            n_exz = self._check_n(self.spatial_basis_set.expand_coeffs(self.n_ex_array, 0, 0, z))

            axs[0].plot(x, nx, c='k', marker='o', label=r'$n_{total}$')
            axs[0].plot(x, n0x, c='b', marker='o', label=r'$n_0$')
            axs[0].plot(x, n_exx, c='g', marker='o', label=r'$n_{ex}$')
            axs[0].set_title(r'$n(x,0,0)$', fontsize=18)
            axs[0].set_xlabel(r'$x \; \left[\mu m\right]$', fontsize=14)
            axs[0].set_ylabel(r'$n(x,0,0) \; \left[\mu m^{-3}\right]$', fontsize=14)

            axs[1].plot(y, ny, c='k', marker='o', label=r'$n_{total}$')
            axs[1].plot(y, n0y, c='b', marker='o', label=r'$n_0$')
            axs[1].plot(y, n_exy, c='g', marker='o', label=r'$n_{ex}$')
            axs[1].set_title(r'$n(0,y,0)$', fontsize=18)
            axs[1].set_xlabel(r'$y \; \left[\mu m\right]$', fontsize=14)
            axs[1].set_ylabel(r'$n(0,y,0) \; \left[\mu m^{-3}\right]$', fontsize=14)

            axs[2].plot(z, nz, c='k', marker='o', label=r'$n_{total}$')
            axs[2].plot(z, n0z, c='b', marker='o', label=r'$n_0$')
            axs[2].plot(z, n_exz, c='g', marker='o', label=r'$n_{ex}$')
            axs[2].set_title(r'$n(0,0,z)$', fontsize=18)
            axs[2].set_xlabel(r'$z \; \left[\mu m\right]$', fontsize=14)
            axs[2].set_ylabel(r'$n(0,0,z) \; \left[\mu m^{-3}\right]$', fontsize=14)

            for i in range(3):
                ax2 = axs[i].twinx()  # Create a secondary y-axis for potential
                if i == 0:
                    line1 = ax2.plot(x, self.particle_props.V_trap(x, 0, 0), 'r--', label=r'$V_{trap}$')  
                elif i == 1:
                    ax2.plot(y, self.particle_props.V_trap(0, y, 0), 'r--', label=r'$V_{trap}$')
                elif i == 2:
                    ax2.plot(z, self.particle_props.V_trap(0, 0, z), 'r--', label=r'$V_{trap}$')
                
                ax2.set_ylabel(r'$V_{trap} \; \left[ nK \right]$', color='r', fontsize=14)  
                ax2.tick_params(axis='y', labelcolor='r')  

                axs[i].grid(True)
                
            h, _ = axs[0].get_legend_handles_labels()  
            labels = [r'$n_{total}$', r'$n_0$', r'$n_{ex}$', '$V_{trap}$']
            fig.legend(h+line1, labels, loc='upper right', fontsize=14, fancybox=True, framealpha=0.9, bbox_to_anchor=(1, 1))  

            fig.tight_layout(rect=[0, 0, 0.95, 1])
            if filename != None:
                fig.savefig(filename, dpi=300, bbox_inches='tight')

            return fig
        
        else:
            print("No convergence history found. Please run eval_density() first.")


    def plot_density_2d(
            self, 
            which: str = 'all', 
            num_points: int = 200,
            **kwargs,
    ) -> Figure:
        """Plot the spatial density n(x,y,0), n(x,0,z) and n(0,y,z) along two directions respectively.
        
           Args:
                which: which densities to plot, either 'all', 'n', 'n0', or 'n_ex'. Defaults to 'all'.
                num_points: number of points along each direction to evaluate the density at. Defaults to 200.
                **kwargs: keyword arguments for the plot, supported are 'title' and 'filename'.

           Returns:
                fig: matplotlib figure object of the plot
        """
        if (self.particle_props.T.value < self.zero_T_threshold and self.N_particles is not None) or np.linalg.norm(self.n_ex_array) > 0:
            title = kwargs.get('title', 'Spatial density, T='+str(self.particle_props.T)+', N='+str(int(self.N_particles)))
            filename = kwargs.get('filename', None)

            # Determine the number of rows for the plot
            nrows = 1 if which != 'all' else 3
            fig = plt.figure(figsize=(17, 5 * nrows))
            gs = gridspec.GridSpec(nrows, 4, width_ratios=[1, 1, 1, 0.05])
            gs.update(wspace=0.65)
            fig.suptitle(title, fontsize=24, y=0.94)

            x = np.linspace(self.particle_props.domain[0][0].value, self.particle_props.domain[0][1].value, num_points)
            y = np.linspace(self.particle_props.domain[1][0].value, self.particle_props.domain[1][1].value, num_points)
            z = np.linspace(self.particle_props.domain[2][0].value, self.particle_props.domain[2][1].value, num_points)

            # Function to create plots for a given density array
            def create_plots(density_array, row_idx, density_label):
                for j in range(3):
                    ax = plt.subplot(gs[row_idx, j])
                    # Determine the 2D slice based on the subplot
                    if j == 0:  # x-y plane
                        X, Y = np.meshgrid(x, y)
                        slice_2d = self._check_n(self.spatial_basis_set.expand_coeffs(density_array, X, Y, 0))
                        im = ax.pcolormesh(x, y, slice_2d)
                        ax.set_aspect('equal', adjustable='box')
                        ax.set_xlabel('x [μm]', fontsize=12)
                        ax.set_ylabel('y [μm]', fontsize=12)
                        title = f'{density_label}(x, y, 0)'
                    elif j == 1:  # x-z plane
                        X, Z = np.meshgrid(x, z)
                        slice_2d = self._check_n(self.spatial_basis_set.expand_coeffs(density_array, X, 0, Z))
                        im = ax.pcolormesh(x, z, slice_2d)
                        ax.set_aspect('equal', adjustable='box')
                        ax.set_xlabel('x [μm]', fontsize=12)
                        ax.set_ylabel('z [μm]', fontsize=12)
                        title = f'{density_label}(x, 0, z)'
                    else:  # y-z plane
                        Y, Z = np.meshgrid(y, z)
                        slice_2d = self._check_n(self.spatial_basis_set.expand_coeffs(density_array, 0, Y, Z))
                        im = ax.pcolormesh(y, z, slice_2d)
                        ax.set_aspect('equal', adjustable='box')
                        ax.set_xlabel('y[μm]', fontsize=12)
                        ax.set_ylabel('z [μm]', fontsize=12)
                        title = f'{density_label}(0, y, z)'
                    ax.set_aspect('equal', adjustable='box')

                    # Plotting
                    ax.set_title(title, fontsize=18)

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = fig.colorbar(im, cax=cax)
                    cbar.ax.set_title(str(density_label) + r' $\left[ \mu m^{-3} \right]$', pad=6, fontsize=12, loc='left')

            # Plotting based on 'which' parameter
            row_idx = 0
            if which in ['all', 'n']:
                create_plots(self.n_array, row_idx, r'$n$')
                row_idx += 1
            if which in ['all', 'n0']:
                create_plots(self.n0_array, row_idx, r'$n_0$')
                row_idx += 1
            if which in ['all', 'n_ex']:
                create_plots(self.n_ex_array, row_idx, r'$n_{ex}$')

            if filename is not None:
                fig.savefig(filename, dpi=300, bbox_inches='tight')

            plt.show()

            return fig

        else:
            print("No convergence history found. Please run eval_density() first.")

   
    def plot_all(
            self,
            which: str = 'all',
    ) -> None:
        """Plot convergence history and all densities.
        
           Args:
                which: which densities to plot, either 'all', 'n', 'n0', or 'n_ex'. Defaults to 'all'.
        """
        a = self.plot_convergence_history()
        b = self.plot_density_1d()
        c = self.plot_density_2d(which)


    def _check_n(
            self,
            n: Union[np.ndarray, Quantity],
    ) -> np.ndarray:
        """Check if the density `n` is in units of 1/um**3 and convert to unitless if so.
        
           Args:
                n: density to be checked

           Returns:
                n: density in unitless form
        """
        if isinstance(n, Quantity):
            if n.unit.is_equivalent(1/u.um**3):
                n = n.to(1/u.um**3).value
            else:
                raise ValueError("n has to be in units equivalent to 1/um**3 if it is a Quantity.")
        return n


    def _check_and_process_input(
            self,
            which_method: str,
    ) -> None:
        """Check and process the input arguments for selected methods.

           Args:
                which_method: string indicating which method is called, either 'init' or 'eval_density'
        """
        if which_method == "init":
            # Initialize particle properties
            if isinstance(self.particle_props, ParticleProps):
                if self.particle_props.species != 'boson':
                    raise ValueError('particle_props must be a boson ParticleProps object')
            else:
                raise TypeError('particle_props must be a ParticleProps object')


            if isinstance(self.spatial_basis_set, SpatialBasisSet):
                if self.spatial_basis_set.domain != self.particle_props.domain:
                    self.spatial_basis_set.domain = self.particle_props.domain
                    print("WARNING: spatial_basis_set domain was set to particle_props.domain.")
            elif isinstance(self.spatial_basis_set, str):
                if self.spatial_basis_set == 'grid':
                    num_grid_points = self.basis_set_kwargs.get('num_grid_points', 101) 
                    potential_function = self.basis_set_kwargs.get('potential_function', self.particle_props.V_trap)
                    self.spatial_basis_set = GridSpatialBasisSet(
                        self.particle_props.domain, 
                        num_grid_points,
                        potential_function,
                        )
                else:
                    raise NotImplementedError("Only 'grid' is implemented so far.")
            else:
                raise TypeError("spatial_basis_set must be a SpatialBasisSet object or a string.")
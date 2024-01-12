import numpy as np
from scipy.integrate import simpson
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.units as u
from astropy.constants import hbar, k_B
from astropy.units import Quantity
from typing import Union, Sequence

from .particle_props import ParticleProps

class BoseGas:
    """
    The BEC (Bose-Einstein Condensate) class is designed to model and analyze the properties of a Bose-Einstein 
    condensate. It allows for the calculation of the spacial density of weakly interacting bosons at low temperatures
    by iteratively updating the condensed density (either with Thomas Fermi approximation or by solving the 
    generalized Gross-Pitaevskii equation) and the thermal density (either with a semiclassical Hartree Fock or 
    semiclassical Popov approximation). It also fixes the chemical potential to ensure the normalization to a 
    specific particle number.

    Attributes:
        particle_props (ParticleProps): Instance of ParticleProps class containing all relevant particle properties
                                        such as `species`, `m`, `N_particles`, `T`, `domain`, `a_s` and `g` as well
                                        as a wrapper method to call the `V_trap` potential.
        num_grid_points (tuple): Number of grid points in each spatial dimension.
        x, y, z (Quantity): 1D arrays containing the spatial grid points along each direction within the `domain` in [um].
        dx, dy, dz (Quantity): Spacing between grid points along each direction in [um].
        X, Y, Z (np.ndarray): 3D arrays containing the spatial grid points along each direction within the `domain` in [um].
        V_trap_array (np.ndarray): 3D array containing the external trapping potential at each grid point in [nK].
        mu (Quantity): Chemical potential in [nK].
        n0_array (Quantity): 3D array containing the condensed density at each grid point in [um^-3].
        n_ex_array (Quantity): 3D array containing the thermal density at each grid point in [um^-3].
        n_array (Quantity): 3D array containing the total density at each grid point in [um^-3].
        N_particles (Quantity): Total number of particles in the system.
        N_particles_condensed (Quantity): Number of particles in the condensate.
        N_particles_thermal (Quantity): Number of particles in the thermal cloud.
        condensate_fraction (float): Fraction of particles in the condensate.
        convergence_history_mu (list): List containing the chemical potential at each iteration of the iterative procedure.
        convergence_history_N (list): List containing the total particle number at each iteration of the iterative procedure.


    Methods:
        eval_density(self, ...): Run the iterative procedure to update densities and find chemical potential.
        _update_n0_with_TF_approximation(self): Apply the Thomas-Fermi approximation to update the condensed density.
        _update_n0_by_solving_generalized_GPE(self): Solve the generalized GPE to update the condensed density. (Not implemented yet)
        _update_n_ex(self, ...): Apply semiclassical approximations to update the non-condensed density.
        _integrand_HF(self, ...): Calculate the integrand for the semiclassical Hartree-Fock approximation.
        _integrand_Popov(self, ...): Calculate the integrand for the semiclassical Popov approximation.
        plot_convergence_history(self, ...): Plot the convergence history of the iterative density evaluation procedure.
        plot_density_1d(self, ...): Plot the spatial density along each direction in 1D.
        plot_density_2d(self, ...): Plot the spatial density along two directions in 2D.

    Note:
        The class uses units from the `astropy.units` module for physical quantities.
        It is important to ensure that the input parameters are given in the correct units or are appropriately converted.
    """
    def __init__(
            self, 
            particle_props: ParticleProps,
            num_grid_points: Union[int, Sequence[int], np.ndarray] = [101, 101, 101],
            init_with_zero_T: bool = True,
        ):
        """Initialize BEC class. Make sure to use the correct units as specified below!!! If you are 
           using astropy units, you can use units that are equilvalent to the ones specified below.
        
           Args:
               particle_props: Instance of ParticleProps class containing all relevant particle properties.
               num_grid_points (Sequence, np.ndarray, or int): Number of grid points in each spatial dimension. Either
                                                               a sequence of length 3 containing the number of grid points
                                                               in each dimension, or a single integer specifying the same
                                                               number of grid points in each dimension. Defaults to 101.
               init_with_zero_T (bool): If True, run a zero-temperature calculation to improve initial guess for `mu`.
        """
        # Initialize particle properties
        if isinstance(particle_props, ParticleProps):
            if particle_props.species == 'boson':
                self.particle_props = particle_props
            else:
                raise ValueError('particle_props must be a boson ParticleProps object')
        else:
            raise TypeError('particle_props must be a ParticleProps object')

        # Initialize the spacial grid
        if isinstance(num_grid_points, int):
            self.num_grid_points = (num_grid_points, num_grid_points, num_grid_points)
        elif isinstance(num_grid_points, (Sequence, np.ndarray)):
            if len(num_grid_points) != 3:
                raise ValueError("num_grid_points must be a sequence of length 3.")
            if all(isinstance(n, int) for n in num_grid_points):
                self.num_grid_points = tuple(num_grid_points)
            else:
                raise ValueError("num_grid_points must be a sequence of integers.")
        else:
            raise ValueError("num_grid_points must be an integer or a sequence of integers.")
        self.x = np.linspace(self.particle_props.domain[0][0], self.particle_props.domain[0][1], self.num_grid_points[0]) 
        self.y = np.linspace(self.particle_props.domain[1][0], self.particle_props.domain[1][1], self.num_grid_points[1])
        self.z = np.linspace(self.particle_props.domain[2][0], self.particle_props.domain[2][1], self.num_grid_points[2])
        self.dx = self.x[1] - self.x[0] # TODO: Generalize this to work for non-equidistant grids!
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij',)

        # Initialize the external trapping potential
        self.V_trap_array = self.particle_props.V_trap(self.X.value, self.Y.value, self.Z.value)
        if isinstance(self.V_trap_array, Quantity):
            self.V_trap_array = self.V_trap_array.to(u.nK)
        else:
            self.V_trap_array *= u.nK
        
        # Initilize the chemical potential using the TF approximation. In principle this would give:
        # mu(r) = V(r) + g*(n0(r) + 2*n_ex(r))
        # But we don't want a position-dependent mu, so as an initial guess we take:
        self.mu = np.min(self.V_trap_array) + self.particle_props.g.value*(self.particle_props.N_particles**(1/3))*u.nK 

        # Initialize the densities and particle numbers, which get assigned meaningful values after running 
        # eval_density()
        self.n0_array = None
        self.n_ex_array = None
        self.n_array = None
        self.N_particles = None
        self.N_particles_condensed = None
        self.N_particles_thermal = None
        self.condensate_fraction = None

        # Run a zero-temperature calculation to improve initial guess for `mu`. If the provided temperature is 
        # already zero, we don't run this in order to not disturb the workflow of always running eval_density() 
        # after initializing this class.
        if init_with_zero_T and self.particle_props.T.value > 1e-3:
            T = self.particle_props.T
            self.particle_props.T = 0 * u.nK 
            self.eval_density(use_TF=True, use_Popov=False, show_progress=False)       
            self.particle_props.T = T
        

    def eval_density(
            self,
            use_TF: bool = True,
            use_Popov: bool = False,
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
        """Run the iterative procedure to update in turn the densities `n_0_array` and `n_ex_array`. 
        The procedure is repeated until the chemical potential is converged or the maximum number 
        of iterations is reached. After running this method, the total density `n_array`, condensed 
        density `n0_array` and non-condensed density `n_ex_array` have meaningful values if the 
        convergence criterion was met. You can run `plot_convergence_history()` to see if you are 
        satisfied with the convergence.
        
           Args:
               use_TF: if True, use the Thomas-Fermi approximation to update n0, 
                       if False, the generalized GPE is solved numerically to update n0
                       (might be computationally intensive). Defaults to True.
               use_Popov: if True, use the semiclassical Popov approximation to update n_ex,
                          if False, use the semiclassical Hartree-Fock approximation to update n_ex.
                          Defaults to False. TODO: Fix division by zero problem in Popov implementation!
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
               show_progress: if True, use tqdm progress bar for for iterative algorithm and print out the iteration after
                              which convergence was reached, if False, don't use progress bar and don't print out after 
                              which iteration convergence was reached. Defaults to True.
        """
        # Approximations
        self.use_TF = use_TF
        self.use_Popov = use_Popov

        # Set up convergence history list for the plot_convergence_history() method
        self.convergence_history_mu = [self.mu.value]
        self.convergence_history_N = [self.N_particles]
            
        # Run iterative procedure to update the densities
        iterator = tqdm(range(max_iter)) if show_progress else range(max_iter)
        for iteration in iterator: 
            # Initialize n_ex_array with zeros in first iteration
            if iteration == 0:
                    self.n_ex_array = np.zeros(self.num_grid_points) * 1/u.um**3

            # Update condensed density n0
            if self.use_TF:
                self._update_n0_with_TF_approximation()
            else:
                raise NotImplementedError("This method is not implemented yet.")
            
            # Update non-condensed density n_ex if T>0 (otherwise we have n_ex=0)
            if self.particle_props.T.value > 1e-3:
                self._update_n_ex(num_q_values, cutoff_factor) # This uses either the semiclassical HF or Popov approximation
                                                               # depending on the self.use_Popov flag.
            
            # Update the total density n = n0 + nex
            self.n_array = self.n0_array + self.n_ex_array

            # Update the particle numbers
            self.N_particles = np.sum(self.n_array) * self.dx*self.dy*self.dz
            self.N_particles_condensed = np.sum(self.n0_array) * self.dx*self.dy*self.dz
            self.N_particles_thermal = np.sum(self.n_ex_array) * self.dx*self.dy*self.dz
            self.condensate_fraction = self.N_particles_condensed / self.N_particles

            # Do soft update of the chemical potential mu based on normalization condition int dV (n0 + nex) = N_particles.
            # This increases mu, if N_particles is too small w.r.t N_particles_target and decreases mu if N_particles is
            # too large w.r.t. N_particles_target.
            new_mu_direction = (self.particle_props.N_particles - self.N_particles) / self.particle_props.N_particles * u.nK 
            self.mu += mu_change_rate * new_mu_direction

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
                    print('delta_mu: ', delta_mu_value)
                    print('new_mu_direction: ', new_mu_direction)
                    print('mu_change_rate: ', mu_change_rate)
                    print("\n")

            # Dynamically adjust `mu_change_rate` based on recent changes
            if iteration % mu_change_rate_adjustment == 0 and iteration > 4:
                # Check for oscillations in mu
                oscillating = False
                for i in range(1, 5): # Check if mu was oscillating over last 4 iterations
                    if  (self.convergence_history_mu[-i] - self.convergence_history_mu[-i-1]) * \
                        (self.convergence_history_mu[-i-1] - self.convergence_history_mu[-i-2]) < 0:
                        oscillating = True
                        break
                if oscillating: # If oscillating, decrease the change rate to stabilize
                    mu_change_rate *= 0.5 
                else: # If not oscillating, increase the change rate to speed up convergence
                    mu_change_rate *= 2 


            # Check convergence criterion
            if delta_mu_value < mu_convergence_threshold*np.abs(self.mu.value) and \
               np.abs(self.particle_props.N_particles-self.N_particles) < N_convergence_threshold*self.particle_props.N_particles:
                if show_progress:
                    print(f"Convergence reached after {iteration} iterations.")
                break
                    

    def _update_n0_with_TF_approximation(
            self,
        ):
        """Apply the Thomas-Fermi approximation to update the condensed density `n_0_array`.
           When μ>V(r)+2*g*nex(r), the expression for n0(r) is positive, indicating the presence of 
           condensed particles at that location.
           However, if μ≤V(r)+2*g*nex(r), it implies that the local chemical potential is not sufficient 
           to support a condensate at that point due to either the external potential V(r) being too high 
           or the interaction energy with the non-condensed particles being too significant. 
           In this case, the density of condensed particles n0(r) should be zero, as having a negative 
           density is physically meaningless.
        """
        # Calculate n0 with current chemical potential 
        #mask = self.mu > (self.V_trap_array + 2*self.particle_props.g*self.n_ex_array)
        #self.n0_array = np.zeros_like(self.V_trap_array) 
        #self.n0_array[mask] = (self.mu - (self.V_trap_array + 2*self.particle_props.g*self.n_ex_array))[mask] / self.particle_props.g
        self.n0_array = np.maximum((self.mu - (self.V_trap_array + 2*self.particle_props.g*self.n_ex_array)) / self.particle_props.g, 0)


    def _update_n0_by_solving_generalized_GPE(self,):
        """Solve the generalized GPE to update the condensed density `n0_array`."""
        raise NotImplementedError("This method is not implemented yet.")


    def _update_n_ex(
            self,
            num_q_values: int = 50,
            cutoff_factor: float = 10,
        ):
        """Apply the semiclassical Hartree-Fock or Popov approximation to update the non-condensed 
           density `n_ex_array`. 
           
           Args:
                num_q_values: number of integrand evaluations for the simpson rule to integrate over momentum space. 
                              Defaults to 50.
        """
        # Since we are at low temperature, integrating to infinite momentum is not necessary
        # and will only lead to numerical problems since our excited particles have very low p
        # and numerically we can only sum over a finite set of integrand evaluations and very
        # likely just skip the region of interest (low p) und get out 0 from the integration.
        # Thus we set a cutoff at p_cutoff = sqrt(2*pi*m*k*T) which corresponds to the momentum
        # scale of the thermal de Broglie wavelength (i.e. lambda_dB = h/p_cutoff).
        p_cutoff = np.sqrt(cutoff_factor * 2*np.pi*self.particle_props.m*k_B*self.particle_props.T).to(u.u*u.m/u.s) 
        # use this momentum units to deal with numerical numbers roughly on the order ~1

        # Also, for increased numerical stability, we can rescale the integral to the interval 
        # [0,1] and integrate over the dimensionless variable q = p/p_cutoff instead of p.
        q_values = np.linspace(0, 1, num_q_values) 
        q_values = q_values[:, np.newaxis, np.newaxis, np.newaxis] # reshape to broadcast with spacial grid later 

        # Integrate using Simpson's rule (I chose this over quad() because quad() only works for scalar
        # integrands, but we have a 3d array of integrand values. So to use quad() we would need to loop
        # over the spatial grid and call quad() for each grid point, which is very slow. Simpson() can
        # integrate over the whole array at once in a vectorized way, which is much faster.)
        if self.use_Popov:
            integrand_values = self._integrand_Popov(q_values, p_cutoff)
        else:
            integrand_values = self._integrand_HF(q_values, p_cutoff)

        # Check if integrand is zero at q=1 and integrate over q in the interval [0,1]
        max_integrand_val = np.max(integrand_values)
        max_integrand_val_at_q1 = np.max(integrand_values[-1,:,:,:])
        if max_integrand_val_at_q1 > 1e-10 * max_integrand_val: # Check if integrand is not zero at q=1
            print("WARNING: Integrating until q=1, but integrand is not zero at q=1. Consider increasing cutoff_factor.")
        integral = simpson(integrand_values, q_values.flatten(), axis=0)
#
        # Update n_ex_array 
        self.n_ex_array = np.maximum((integral*(p_cutoff.unit)**3 / (2*np.pi * hbar)**3).to(1/u.um**3), 0)
        #return self._integrand_HF(q_values, p_cutoff)


    def _integrand_HF(
            self, 
            q: Union[float, np.ndarray],
            p_cutoff: Quantity,
        ) -> np.ndarray:
        """Calculate the integrand in eq. 8.121 in "Bose Einstein Condensation in Dilute Gases" by C.J. Pethick 
           and H. Smith. Eq. 8.121 is a 3d volume-integral in momentum space, but luckily only depends on the 
           magnitude of the momentum, so we can integrate over the angles analytically giving us the required 
           integrand as 4*pi*p^2*f(eps_p-mu). Since we rescale the integral to integrate over the dimensionless 
           variable q = p/p_cutoff, we additionally need to multiply the integrand by p_cutoff.
        
           Args:
                q: dimensionless integration variable defined as q = p/p_cutoff
                p_cutoff: cutoff momentum for the integration
                
            Returns:
                f: Integrand p_cutoff*4*pi*p^2*f(eps_p) for the integration over q in the interval [0,1]
        """
        p = q * p_cutoff

        # Calculation of eps_p(r) in semiclassical HF approximation (eq. 8.115). Note, that even if we use 
        # the TF approximation for n0, we still need to incorporate the kinetic energy term for the excited states!
        eps_p = (p**2/(2*self.particle_props.m) / k_B).to(u.nK) + self.V_trap_array + 2*self.particle_props.g*(self.n0_array + self.n_ex_array)

        return p_cutoff.value*4*np.pi*p.value**2 / (np.exp((eps_p-self.mu) / self.particle_props.T) - 1) 



    def _integrand_Popov(
            self, 
            q: Union[float, np.ndarray],
            p_cutoff: Quantity,
        ) -> np.ndarray:
        #TODO: Somehow make sure that there is no divion by zero... (happens if chemical potential is too high
        #      such that eps_p becomes zero. 
        """Calculate the integrand in eq. 8.122 in "Bose Einstein Condensation in Dilute Gases" by C.J. Pethick 
           and H. Smith. Eq. 8.122 is a 3d volume-integral in momentum space, but luckily only depends on the 
           magnitude of the momentum, so we can integrate over the angles analytically giving us the required 
           integrand as 4*pi*p^2*((p^2/2m + 2gn_array + V_array -mu)/eps_p)*f(eps_p). Since we rescale the integral 
           to integrate over the dimensionless variable q = p/p_cutoff, we additionally need to multiply the 
           integrand by p_cutoff.
        
           Args:
                q: dimensionless integration variable defined as q = p/p_cutoff
                p_cutoff: cutoff momentum for the integration
                
            Returns:
                f: Integrand p_cutoff*4*pi*p^2*((p^2/2m + 2gn_array + V_array -mu)/eps_p)*f(eps_p) for the 
                   integration over q in the interval [0,1]
        """
        p = q * p_cutoff

        # Calculation of eps_p(r) in semiclassical Popov approximation (eq. 8.119). Note, that even if we use 
        # the TF approximation for n0, we still need to incorporate the kinetic energy term for the excited states!
        eps_p = np.sqrt(np.maximum(((p**2/(2*self.particle_props.m) / k_B).to(u.nK) + self.V_trap_array + \
                        2*self.particle_props.g*(self.n0_array + self.n_ex_array) - self.mu)**2 - (self.particle_props.g*self.n0_array)**2, 0))

        num_non_condensed = ((p**2/(2*self.particle_props.m) / k_B).to(u.nK) + self.V_trap_array + \
                              2*self.particle_props.g*(self.n0_array + self.n_ex_array) - self.mu) / eps_p
        
        return p_cutoff.value*4*np.pi*p.value**2 * num_non_condensed / (np.exp(eps_p / self.particle_props.T) - 1)


    def plot_convergence_history(
            self,
            start: int = 0,
            end: int = -1,
            **kwargs,
        ):
        """Plot the convergence history of the iterative procedure carried out in `eval_density()`.

           Args:
                start: start index of the convergence history list to be plotted. Defaults to 0.
                end: end index of the convergence history list to be plotted. Defaults to -1, i.e. last index.
        """
        if (self.particle_props.T.value < 1e-3 and self.N_particles is not None) or np.linalg.norm(self.n_ex_array) > 0:
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

            return axs
        
        else:
            print("No convergence history found. Please run eval_density() first.")


    def plot_density_1d(
            self, 
            **kwargs,
        ):
        """Plot the spacial density n(x,0,0), n(0,y,0) and n(0,0,z) along each direction respectively."""
        if (self.particle_props.T.value < 1e-3 and self.N_particles is not None) or np.linalg.norm(self.n_ex_array) > 0:
            title = kwargs.get('title', 'Spacial density, T='+str(self.particle_props.T)+', N='+str(int(self.N_particles)))  
            filename = kwargs.get('filename', None) 

            fig, axs = plt.subplots(1, 3, figsize=(14, 6))
            plt.subplots_adjust(top=0.85)
            fig.suptitle(title, fontsize=24)

            axs[0].plot(self.x, self.n_array[:, self.num_grid_points[1]//2, self.num_grid_points[2]//2], c='k', marker='o', label=r'$n_{total}$')
            axs[0].plot(self.x, self.n0_array[:, self.num_grid_points[1]//2, self.num_grid_points[2]//2], c='b', marker='o', label=r'$n_0$')
            axs[0].plot(self.x, self.n_ex_array[:, self.num_grid_points[1]//2, self.num_grid_points[2]//2], c='g', marker='o', label=r'$n_{ex}$')
            axs[0].set_title(r'$n(x,0,0)$', fontsize=18)
            axs[0].set_xlabel(r'$x \; \left[\mu m\right]$', fontsize=14)
            axs[0].set_ylabel(r'$n(x,0,0) \; \left[\mu m^{-3}\right]$', fontsize=14)

            axs[1].plot(self.y, self.n_array[self.num_grid_points[0]//2, :, self.num_grid_points[2]//2], c='k', marker='o', label=r'$n_{total}$')
            axs[1].plot(self.y, self.n0_array[self.num_grid_points[0]//2, :, self.num_grid_points[2]//2], c='b', marker='o', label=r'$n_0$')
            axs[1].plot(self.y, self.n_ex_array[self.num_grid_points[0]//2, :, self.num_grid_points[2]//2], c='g', marker='o', label=r'$n_{ex}$')
            axs[1].set_title(r'$n(0,y,0)$', fontsize=18)
            axs[1].set_xlabel(r'$y \; \left[\mu m\right]$', fontsize=14)
            axs[1].set_ylabel(r'$n(0,y,0) \; \left[\mu m^{-3}\right]$', fontsize=14)

            axs[2].plot(self.z, self.n_array[self.num_grid_points[0]//2, self.num_grid_points[1]//2, :], c='k', marker='o', label=r'$n_{total}$')
            axs[2].plot(self.z, self.n0_array[self.num_grid_points[0]//2, self.num_grid_points[1]//2, :], c='b', marker='o', label=r'$n_0$')
            axs[2].plot(self.z, self.n_ex_array[self.num_grid_points[0]//2, self.num_grid_points[1]//2, :], c='g', marker='o', label=r'$n_{ex}$')
            axs[2].set_title(r'$n(0,0,z)$', fontsize=18)
            axs[2].set_xlabel(r'$z \; \left[\mu m\right]$', fontsize=14)
            axs[2].set_ylabel(r'$n(0,0,z) \; \left[\mu m^{-3}\right]$', fontsize=14)

            for i in range(3):
                ax2 = axs[i].twinx()  # Create a secondary y-axis for potential
                if i == 0:
                    line1 = ax2.plot(self.x, self.V_trap_array[:, self.num_grid_points[1]//2, self.num_grid_points[2]//2], 'r--', label=r'$V_{trap}$')  
                elif i == 1:
                    ax2.plot(self.y, self.V_trap_array[self.num_grid_points[0]//2, :, self.num_grid_points[2]//2], 'r--', label=r'$V_{trap}$')
                elif i == 2:
                    ax2.plot(self.z, self.V_trap_array[self.num_grid_points[0]//2, self.num_grid_points[1]//2, :], 'r--', label=r'$V_{trap}$')
                
                ax2.set_ylabel(r'$V_{trap} \; \left[ nK \right]$', color='r', fontsize=14)  
                ax2.tick_params(axis='y', labelcolor='r')  

                axs[i].grid(True)
                
            h, _ = axs[0].get_legend_handles_labels()  
            labels = [r'$n_{total}$', r'$n_0$', r'$n_{ex}$', '$V_{trap}$']
            fig.legend(h+line1, labels, loc='upper right', fontsize=14, fancybox=True, framealpha=0.9, bbox_to_anchor=(1, 1))  

            fig.tight_layout(rect=[0, 0, 0.95, 1])
            if filename != None:
                fig.savefig(filename, dpi=300, bbox_inches='tight')

            return axs
        
        else:
            print("No convergence history found. Please run eval_density() first.")



    def plot_density_2d(self, which: str = 'all', **kwargs):
        """Plot the spatial density n(x,y,0), n(x,0,z) and n(0,y,z) along two directions respectively.
        
        Args:
                which: which densities to plot, either 'all', 'n', 'n0', or 'n_ex'. Defaults to 'all'."""

        if (self.particle_props.T.value < 1e-3 and self.N_particles is not None) or np.linalg.norm(self.n_ex_array) > 0:
            title = kwargs.get('title', 'Spatial density, T='+str(self.particle_props.T)+', N='+str(int(self.N_particles)))
            filename = kwargs.get('filename', None)

            # Determine the number of rows for the plot
            nrows = 1 if which != 'all' else 3
            fig = plt.figure(figsize=(17, 5 * nrows))
            gs = gridspec.GridSpec(nrows, 4, width_ratios=[1, 1, 1, 0.05])
            gs.update(wspace=0.65)
            fig.suptitle(title, fontsize=24, y=0.94)

            # Function to create plots for a given density array
            def create_plots(density_array, row_idx, density_label):
                for j in range(3):
                    ax = plt.subplot(gs[row_idx, j])
                    # Determine the 2D slice based on the subplot
                    if j == 0:  # x-y plane
                        slice_2d = density_array[:, :, self.num_grid_points[2]//2].value
                        im = ax.imshow(slice_2d, extent=[self.x[0].value, self.x[-1].value, self.y[0].value, self.z[-1].value], \
                                       origin='lower')
                    elif j == 1:  # x-z plane
                        slice_2d = density_array[:, self.num_grid_points[1]//2, :].value
                        im = ax.imshow(slice_2d, extent=[self.x[0].value, self.x[-1].value, self.z[0].value, self.z[-1].value], \
                                       origin='lower')
                    else:  # y-z plane
                        slice_2d = density_array[self.num_grid_points[0]//2, :, :].value
                        im = ax.imshow(slice_2d, extent=[self.y[0].value, self.y[-1].value, self.z[0].value, self.z[-1].value], \
                                       origin='lower')

                    # Plotting
                    ax.set_title(f'${density_label}({", ".join("xyz"[k] if k == j else "0" for k in range(3))})$', fontsize=18)
                    ax.set_xlabel(f'{["y", "z", "x"][j]} [μm]', fontsize=12)
                    ax.set_ylabel(f'{["x", "y", "z"][j]} [μm]', fontsize=12)

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

        else:
            print("No convergence history found. Please run eval_density() first.")

   
    def plot_all(
            self,
            which: str = 'all',
    ):
        """Plot all convergence history and all densities.
        
           Args:
                which: which densities to plot, either 'all', 'n', 'n0', or 'n_ex'. Defaults to 'all'.
        """
        a = self.plot_convergence_history()
        b = self.plot_density_1d()
        c = self.plot_density_2d(which)
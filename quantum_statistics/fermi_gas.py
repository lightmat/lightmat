import numpy as np
import sparse
from scipy.integrate import simpson
from scipy.optimize import minimize
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.units as u
from astropy.constants import hbar, k_B
from astropy.units import Quantity
from typing import Union, Sequence

from .particle_props import ParticleProps
from .spatial_basis import SpatialBasisSet, GridSpatialBasisSet

class FermiGas:
    """
    The FermiGas class is designed to model and analyze the properties of a Fermi Gas. It allows for the 
    calculation of the spatial density of weakly interacting fermions at low temperatures by solving the 
    single-particle eigenvalue problem and applies the Fermi-Dirac distribution. It also fixes the chemical 
    potential to ensure the normalization to a specific particle number.

    Attributes:
        particle_props (ParticleProps): Instance of ParticleProps class containing all relevant particle properties
                                        such as `species`, `m`, `N_particles`, `T`, and `domain`as well as a wrapper 
                                        method to call the `V_trap` potential.
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


    Note:
        The class uses units from the `astropy.units` module for physical quantities.
        It is important to ensure that the input parameters are given in the correct units or are appropriately converted.
    """
    def __init__(
            self, 
            particle_props: ParticleProps,
            spatial_basis_set: Union[SpatialBasisSet, str] = 'grid',
            init_with_zero_T: bool = True,
            **basis_set_kwargs,
        ):
        """Initialize FermiGas class. Make sure to use the correct units as specified below!!! If you are 
           using astropy units, you can use units that are equilvalent to the ones specified below.
        
           Args:
               particle_props: Instance of ParticleProps class containing all relevant particle properties.
               num_grid_points (Sequence, np.ndarray, or int): Number of grid points in each spatial dimension. Either
                                                               a sequence of length 3 containing the number of grid points
                                                               in each dimension, or a single integer specifying the same
                                                               number of grid points in each dimension. Defaults to 101.
               init_with_zero_T (bool): If True, run a zero-temperature calculation to improve initial guess for `mu`.
        """
        self.particle_props = particle_props
        self.spatial_basis_set = spatial_basis_set
        self.basis_set_kwargs = basis_set_kwargs
        self._check_and_process_input("init")
        
        # Initialize the external trapping potential
        self.V_trap_array = self.spatial_basis_set.get_coeffs(self.particle_props.V_trap)
        if isinstance(self.V_trap_array, Quantity):
            self.V_trap_array = self.V_trap_array.to(u.nK)
        else:
            self.V_trap_array *= u.nK
        
        # Initilize the chemical potential using the TF approximation. In principle, this would give:
        # mu = hbar^2 k_F(r)^2 / 2m + V_trap(r)
        # But we don't want a position-dependent mu, so as an initial guess we take:
        self.mu = np.min(self.V_trap_array) + np.abs(np.average(self.V_trap_array))

        # Initialize the density arrays
        self.n_array = None
        self.N_particles = None

        # Run a zero-temperature calculation to improve initial guess for `mu`. If the provided temperature is 
        # already zero, we don't run this in order to not disturb the workflow of always running eval_density() 
        # after initializing this class.
        if init_with_zero_T and self.particle_props.T.value > 1e-3:
            T = self.particle_props.T
            self.particle_props.T = 0 * u.nK 
            self.eval_density(use_LDA=True, show_progress=False)       
            self.particle_props.T = T


    def eval_density(
            self,
            use_LDA: bool = True,
            max_iter: int = 1000,
            mu_convergence_threshold: float = 1e-5,
            N_convergence_threshold: float = 1e-3,
            mu_change_rate: float = 0.1,
            mu_change_rate_adjustment: int = 5,
            num_q_values: int = 50,
            cutoff_factor: float = 100,
            print_convergence_info_at_this_iteration: int = 0,
            show_progress: bool = True,
        ):
        """Run the iterative procedure to update the density `n_array`. The procedure is repeated until
           the chemical potential is converged or the maximum number of iterations is reached. You can 
           run `plot_convergence_history()` to see if you are satisfied with the convergence.
        
           Args:
               use_LDA: if True, use the semiclassical LDA to update n, if False, use the Thomas-Fermi 
                        approximation to update n. Defaults to True.
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
                             semiclassically calculate density `n_array`. Defaults to 50.
               cutoff_factor: factor to determine the cutoff momentum for the momentum space integral to semiclassically
                              calculate density `n_array`. The cutoff momentum is given by 
                              p_cutoff = sqrt(cutoff_factor * 2*m*k_B*T). Defaults to 100.
               print_convergence_info_at_this_iteration: if 0, don't print convergence info, if this value is i>0, 
                                                         then print convergence info every i-th iteration. Defaults to 0.
               show_progress: if True, use tqdm progress bar for for iterative algorithm and print out the iteration after
                              which convergence was reached, if False, don't use progress bar and don't print out after 
                              which iteration convergence was reached. Defaults to True.
        """
        # Approximation
        self.use_LDA = use_LDA

        # Set up convergence history list for the plot_convergence_history() method
        self.convergence_history_mu = [self.mu.value]
        self.convergence_history_N = [self.N_particles]
            
        # Run iterative procedure to update the densities
        iterator = tqdm(range(max_iter)) if show_progress else range(max_iter)
        for iteration in iterator: 
            # Update density n
            if self.particle_props.T.value < 1e-3:
                self._update_n_with_TF_approximation()
            elif self.use_LDA:
                self._update_n_with_LDA(num_q_values, cutoff_factor)
            else:
                self._update_n_with_E_functional_minimization()
            
            # Update the particle number
            self.N_particles = self.spatial_basis_set.integral(self.n_array)

            # Do soft update of the chemical potential mu based on normalization condition int dV n = N_particles.
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


    def _update_n_with_TF_approximation(
            self,
        ):
        """Update the density `n_array` using the Thomas-Fermi approximation. The density is zero where the
           chemical potential is smaller than the external trapping potential. Else, the density is given by
           the Thomas-Fermi approximation.
        """
        mask = (self.mu - self.V_trap_array) >= 0
        self.n_array = np.zeros_like(self.V_trap_array.value)
        self.n_array[mask] = (1/(6*np.pi**2) * ((2*self.particle_props.m/(hbar**2) * k_B) * \
                                     (self.mu - self.V_trap_array[mask]))**(3/2)).to(1/u.um**3).value
        self.n_array = self.n_array * 1/u.um**3



    def _update_n_with_LDA(
            self,
            num_q_values: int = 50,
            cutoff_factor: float = 100,
        ):
        """Update the density `n_array` using the local density approximation. This means unsing a semiclassical
           approximation to integrate the Fermi-Dirac distribution over momentum space, where we insert for the
           single-particle energy the classical expression eps(p,r) = p^2/2m + V_trap(r)."""
        # Since we are at low temperature, integrating to infinite momentum is not necessary
        # and will only lead to numerical problems since our excited particles have very low p
        # and numerically we can only sum over a finite set of integrand evaluations and very
        # likely just skip the region of interest (low p) und get out 0 from the integration.
        # Thus we set a cutoff at p_cutoff = sqrt(cutoff_factor * 2*m*k_B*T) which corresponds to 
        # an energy scale of p_cutoff^2/(2*m) = cutoff_factor * k_B*T. This is a good approximation
        # since the Fermi-Dirac distribution is very steep and the integral is dominated by the
        # regions around the Fermi energy, which is of the order of k_B*T.
        p_cutoff = np.sqrt(cutoff_factor * 2*self.particle_props.m*k_B*self.particle_props.T).to(u.u*u.m/u.s)
        # use this momentum units to deal with numerical numbers roughly on the order ~1

        # Also, for increased numerical stability, we can rescale the integral to the interval 
        # [0,1] and integrate over the dimensionless variable q = p/p_cutoff instead of p.
        q_values = np.linspace(0, 1, num_q_values) 
        q_values = q_values[:, np.newaxis] # reshape to broadcast with spatial basis set later

        # Integrate using Simpson's rule (I chose this over quad() because quad() only works for scalar
        # integrands, but we have a 3d array of integrand values. So to use quad() we would need to loop
        # over the spatial grid and call quad() for each grid point, which is very slow. Simpson() can
        # integrate over the whole array at once in a vectorized way, which is much faster.)
        integrand_values = self._integrand_FD(q_values, p_cutoff) #sparse.COO.from_numpy(self._integrand_FD(q_values, p_cutoff))

        # Check if integrand is zero at q=1 and integrate over q in the interval [0,1]
        max_integrand_val = np.max(integrand_values)
        max_integrand_val_at_q1 = np.max(integrand_values[-1,:])
        if max_integrand_val_at_q1 > 1e-10 * max_integrand_val: # Check if integrand is not zero at q=1
            print("WARNING: Integrating until q=1, but integrand is not zero at q=1. Consider increasing cutoff_factor.")
        integral = simpson(integrand_values, q_values.flatten(), axis=0)

        # Update n_array
        self.n_array = np.maximum((integral*(p_cutoff.unit)**3 / (2*np.pi * hbar)**3).to(1/u.um**3), 0)


    def _integrand_FD(
            self, 
            q: Union[float, np.ndarray],
            p_cutoff: Quantity,
        ) -> np.ndarray:
        """Calculate the integrand, which is a Fermi Dirac distribution with inserted classical energy 
           eps(p,r) = p^2/2m + V_trap(r). The 3d integral over momentum space becomes a 1d integral over 
           the absolute value of momentum since we can carry out the angular integrals analytically leading
           to an additional factor 4pi*p^2 in our integrand.  Since we rescale the integral to integrate over 
           the dimensionless variable q = p/p_cutoff, we additionally need to multiply the integrand by p_cutoff.
        
           Args:
                q: dimensionless integration variable defined as q = p/p_cutoff
                p_cutoff: cutoff momentum for the integration
                
            Returns:
                f: Integrand p_cutoff*4*pi*p^2*f(eps_p) for the integration over q in the interval [0,1]
        """
        p = q * p_cutoff
        eps_p = (p**2/(2*self.particle_props.m) / k_B).to(u.nK) + self.V_trap_array 
        return p_cutoff.value*4*np.pi*p.value**2 / (np.exp((eps_p-self.mu) / self.particle_props.T) + 1) 
    


    def _update_n_with_E_functional_minimization(
            self,
        ):
        E_min = minimize(self._E_functional, self.n_array, jac=self._E_functional_deriv,)
        self.n_array = E_min.x



    def _E_functional(
            self,
            n_array: np.ndarray,
    )-> float:
        Ekin1 = (6*np.pi**2)**(5/3)/(10*np.pi**2) * self.spatial_basis_set.integral(self.spatial_basis_set.power(n_array, 5/3)) 

        # I assume that n(r) defines a boundary surface in 3d space where it becomes zero and that (grad(n(r)))**2 / n(r)
        # is zero beyond that surface since the gradient becomes zero even at infinitesimal distances beyond the surface.
        # Since the boundary surface is a null set it doesn't contribute to the integral. Thus, we can integrate only 
        # over the region where n(r) is non-zero, i.e. mask out the region where n(r) is zero.
        integrand = np.zeros(self.spatial_basis_set.num_basis_funcs) * 1/u.um**5
        mask = n_array > 0
        integrand[mask] = self.spatial_basis_set.gradient_dot_gradient(n_array)[mask] / n_array[mask]
        Ekin2 = 1/36 * self.spatial_basis_set.integral(integrand)

        Ekin3 = 1/3 * self.spatial_basis_set.integral(self.spatial_basis_set.laplacian(n_array))

        Ekin = (hbar**2/(2*self.particle_props.m) * (Ekin1 + Ekin2 + Ekin3) / k_B).to(u.nK)

        Epot = self.spatial_basis_set.integral(n_array * self.V_trap_array).to(u.nK)

        return (Ekin + Epot)

        

    def _E_functional_deriv(
            self,
            n_array: np.ndarray,
    ):
        grad_Ekin1 = (6*np.pi**2)**(5/3)/(10*np.pi**2) * 5/3 * self.spatial_basis_set.power(n_array, 2/3)

        grad_Ekin2 = np.zeros(self.spatial_basis_set.num_basis_funcs) * 1/u.um**2
        mask = n_array > 0
        grad_Ekin2[mask] = -1/36 * (self.spatial_basis_set.gradient_dot_gradient(n_array)[mask] \
                                / self.spatial_basis_set.power(n_array, 2)[mask] + \
                                2 * self.spatial_basis_set.laplacian(n_array)[mask] / n_array[mask])

        grad_E = (hbar**2/(2*self.particle_props.m) * (grad_Ekin1 + grad_Ekin2) / k_B).to(u.nK)

        return (grad_E + self.V_trap_array)
        



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
        if (self.particle_props.T.value < 1e-3 and self.N_particles is not None) or np.linalg.norm(self.n_array) > 0:
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
            num_points: int = 200,
            **kwargs,
        ):
        """Plot the spatial density n(x,0,0), n(0,y,0) and n(0,0,z) along each direction respectively."""
        if (self.particle_props.T.value < 1e-3 and self.N_particles is not None) or np.linalg.norm(self.n_array) > 0:
            title = kwargs.get('title', 'spatial density, T='+str(self.particle_props.T)+', N='+str(int(self.N_particles)))  
            filename = kwargs.get('filename', None) 

            fig, axs = plt.subplots(1, 3, figsize=(14, 6))
            plt.subplots_adjust(top=0.85)
            fig.suptitle(title, fontsize=24)

            x = np.linspace(self.particle_props.domain[0][0].value, self.particle_props.domain[0][1].value, num_points)
            y = np.linspace(self.particle_props.domain[1][0].value, self.particle_props.domain[1][1].value, num_points)
            z = np.linspace(self.particle_props.domain[2][0].value, self.particle_props.domain[2][1].value, num_points)

            nx = self._check_n(self.spatial_basis_set.expand_coeffs(self.n_array, x, 0, 0))
            ny = self._check_n(self.spatial_basis_set.expand_coeffs(self.n_array, 0, y, 0))
            nz = self._check_n(self.spatial_basis_set.expand_coeffs(self.n_array, 0, 0, z))

            axs[0].plot(x, nx, c='k', marker='o', label=r'$n_{total}$')
            axs[0].set_title(r'$n(x,0,0)$', fontsize=18)
            axs[0].set_xlabel(r'$x \; \left[\mu m\right]$', fontsize=14)
            axs[0].set_ylabel(r'$n(x,0,0) \; \left[\mu m^{-3}\right]$', fontsize=14)

            axs[1].plot(y, ny, c='k', marker='o', label=r'$n_{total}$')
            axs[1].set_title(r'$n(0,y,0)$', fontsize=18)
            axs[1].set_xlabel(r'$y \; \left[\mu m\right]$', fontsize=14)
            axs[1].set_ylabel(r'$n(0,y,0) \; \left[\mu m^{-3}\right]$', fontsize=14)

            axs[2].plot(z, nz, c='k', marker='o', label=r'$n_{total}$')
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
            labels = [r'$n_{total}$', '$V_{trap}$']
            fig.legend(h+line1, labels, loc='upper right', fontsize=14, fancybox=True, framealpha=0.9, bbox_to_anchor=(1, 1))  

            fig.tight_layout(rect=[0, 0, 0.95, 1])
            if filename != None:
                fig.savefig(filename, dpi=300, bbox_inches='tight')

            return axs
        
        else:
            print("No convergence history found. Please run eval_density() first.")


    def plot_density_2d(
            self, 
            num_points: int = 200,
            **kwargs
        ):
        """Plot the spatial density n(x,y,0), n(x,0,z) and n(0,y,z) along two directions respectively."""
        if (self.particle_props.T.value < 1e-3 and self.N_particles is not None) or np.linalg.norm(self.n_array) > 0:
            title = kwargs.get('title', 'Spatial density, T='+str(self.particle_props.T)+', N='+str(int(self.N_particles)))
            filename = kwargs.get('filename', None)

            # Define the figure and GridSpec layout
            fig = plt.figure(figsize=(17, 5))  
            gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])  # 3 columns for plots, 1 for colorbars
            gs.update(wspace=0.65)  

            # Create subplots
            axs = [plt.subplot(gs[0, i]) for i in range(3)]
            fig.suptitle(title, fontsize=24, y=0.94)

            x = np.linspace(self.particle_props.domain[0][0].value, self.particle_props.domain[0][1].value, num_points)
            y = np.linspace(self.particle_props.domain[1][0].value, self.particle_props.domain[1][1].value, num_points)
            z = np.linspace(self.particle_props.domain[2][0].value, self.particle_props.domain[2][1].value, num_points)

            X, Y = np.meshgrid(x, y)
            nxy = self._check_n(self.spatial_basis_set.expand_coeffs(self.n_array, X, Y, 0))

            X, Z = np.meshgrid(x, z)
            nxz = self._check_n(self.spatial_basis_set.expand_coeffs(self.n_array, X, 0, Z))

            Y, Z = np.meshgrid(y, z)
            nyz = self._check_n(self.spatial_basis_set.expand_coeffs(self.n_array, 0, Y, Z))        

            # Plotting n(x,y,0), n(x,0,z), n(0,y,z)
            im = []
            
            im.append(axs[0].pcolormesh(x, y, nxy))
            axs[0].set_title(r'$n(x,y,0)$', fontsize=18)
            axs[0].set_aspect('equal', adjustable='box')

            im.append(axs[1].pcolormesh(x, z, nxz))
            axs[1].set_title(r'$n(x,0,z)$', fontsize=18)
            axs[1].set_aspect('equal', adjustable='box')

            im.append(axs[2].pcolormesh(y, z, nyz))
            axs[2].set_title(r'$n(0,y,z)$', fontsize=18)
            axs[2].set_aspect('equal', adjustable='box')

            # Set labels and colorbars for each plot
            for i, ax in enumerate(axs):
                ax.set_xlabel('y [μm]' if i != 2 else 'z [μm]', fontsize=12)
                ax.set_ylabel('x [μm]' if i != 2 else 'y [μm]', fontsize=12)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(im[i], cax=cax)
                cbar.ax.set_title(r'$n \; \left[ \mu m^{-3} \right]$', pad=6, fontsize=12, loc='left')

            if filename is not None:
                fig.savefig(filename, dpi=300, bbox_inches='tight')

            return axs

        else:
            print("No convergence history found. Please run eval_density() first.")



    def plot_all(
            self,
    ):
        a = self.plot_convergence_history()
        b = self.plot_density_1d()
        c = self.plot_density_2d()


    def _check_n(
            self,
            n: Union[np.ndarray, Quantity],
    ):
        if isinstance(n, Quantity):
            if n.unit.is_equivalent(1/u.um**3):
                n = n.to(1/u.um**3).value
            else:
                raise ValueError("n has to be in units equivalent to 1/um**3 if it is a Quantity.")
        return n


    def _check_and_process_input(
            self,
            which_method: str,
    ):
        if which_method == "init":
            # Initialize particle properties
            if isinstance(self.particle_props, ParticleProps):
                if self.particle_props.species != 'fermion':
                    raise ValueError('particle_props must be a fermion ParticleProps object')
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


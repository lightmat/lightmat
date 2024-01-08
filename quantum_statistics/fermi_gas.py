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

from .particle_props import ParticleProps

class FermiGas:
    """
    The FermiGas class is designed to model and analyze the properties of a Fermi Gas. It allows for the 
    calculation of the spacial density of weakly interacting fermions at low temperatures by solving the 
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
            particle_props: ParticleProps = None,
            m: Union[float, Quantity] = None, 
            N_particles: int = None, 
            T: Union[float, Quantity] = None,
            domain: Union[Sequence[float], Sequence[Sequence[float]], np.ndarray, Quantity] = None,
            num_grid_points: Union[int, Sequence[int], np.ndarray] = [101, 101, 101],
            V_trap: Union[Callable, np.ndarray, Quantity] = None, 
            init_with_zero_T: bool = True,
            **V_trap_kwargs,
        ):
        """Initialize FermiGas class. Make sure to use the correct units as specified below!!! If you are 
           using astropy units, you can use units that are equilvalent to the ones specified below.
        
           Args:
               particle_props: Instance of ParticleProps class containing all relevant particle properties.
                               If particle_props is provided, all other parameters except of `num_grid_points`
                               are ignored.
               species (str): Type of the particle, either 'fermion' or 'boson'.
               m (Quantity or float): Mass of the particle, in [kg].
               N_particles (int): Number of particles in the system.
               T (Quantity or float): Temperature of the system, in [nK].
               domain (Sequence, np.ndarray, or Quantity): Spatial domain of the system. Either a sequence of length 2
                                                           containing the same x,y,z domain, or a sequence of length 3
                                                           containing sequences of length 2 containing the x,y,z domain
                                                           in [um].
               num_grid_points (Sequence, np.ndarray, or int): Number of grid points in each spatial dimension. Either
                                                               a sequence of length 3 containing the number of grid points
                                                               in each dimension, or a single integer specifying the same
                                                               number of grid points in each dimension.
               V_trap (Callable): Function that returns the trapping potential of the system at given position(s).
               name (str): Name of the particle.
               **V_trap_kwargs: Keyword arguments to pass to V_trap.
        """
        # Initialize particle properties
        if particle_props and isinstance(particle_props, ParticleProps):
            self.particle_props = particle_props
        else:
            # Check if all required parameters are provided
            if None in [m, N_particles, T, domain, V_trap]:
                raise ValueError("All parameters must be provided if not using a ParticleProps instance.")
            self.particle_props = ParticleProps(
                name="FermiGas Particle",
                species="fermion",
                m=m,
                N_particles=N_particles,
                T=T,
                domain=domain,
                V_trap=V_trap,
                **V_trap_kwargs,
            )

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
        
        # Initilize the chemical potential using the TF approximation. In principle, this would give:
        # mu = hbar^2 k_F(r)^2 / 2m + V_trap(r)
        # But we don't want a position-dependent mu, so as an initial guess we take:
        #self.mu = np.min(self.V_trap_array) + (hbar**2/(2*self.particle_props.m) / k_B).to(u.nK).value * \
        #                                      (6*np.pi**2 * self.particle_props.N_particles / \
        #                                      ((self.particle_props.V_trap(0,0,0)*self.dx*self.dy*self.dz)).value)**(2/3) * u.nK
            
        self.mu = np.average(self.V_trap_array)

        # Initialize the density arrays
        self.n_array = None
        self.N_particles = None


    def eval_density(
            self,
            use_TF: bool = True,
            max_iter: int = 1000,
            mu_convergence_threshold: float = 1e-5,
            N_convergence_threshold: float = 1e-3,
            mu_change_rate: float = 0.01,
            mu_change_rate_adjustment: int = 5,
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
               print_convergence_info_at_this_iteration: if 0, don't print convergence info, if this value is i>0, 
                                                         then print convergence info every i-th iteration. Defaults to 0.
               show_progress: if True, use tqdm progress bar for for iterative algorithm and print out the iteration after
                              which convergence was reached, if False, don't use progress bar and don't print out after 
                              which iteration convergence was reached. Defaults to True.
        """
        # Approximation
        self.use_TF = use_TF

        # Set up convergence history list for the plot_convergence_history() method
        self.convergence_history_mu = [self.mu.value]
        self.convergence_history_N = [self.N_particles]
            
        # Run iterative procedure to update the densities
        iterator = tqdm(range(max_iter)) if show_progress else range(max_iter)
        for iteration in iterator: 
            # Update density n
            if self.use_TF:
                self._update_n_with_TF_approximation()
            else:
                raise NotImplementedError("This method is not implemented yet.")
            
            # Update the particle number
            self.N_particles = np.sum(self.n_array) * self.dx*self.dy*self.dz

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


    def _update_n_with_TF_approximation(self):
        mask = (self.mu - self.V_trap_array) >= 0
        self.n_array = np.zeros_like(self.V_trap_array.value)
        self.n_array[mask] = (1/(6*np.pi**2) * ((2*self.particle_props.m/(hbar**2) * k_B) * \
                                     (self.mu - self.V_trap_array[mask]))**(3/2)).to(1/u.um**3).value
        self.n_array = self.n_array * 1/u.um**3


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
            axs[0].set_title(r'$n(x,0,0)$', fontsize=18)
            axs[0].set_xlabel(r'$x \; \left[\mu m\right]$', fontsize=14)
            axs[0].set_ylabel(r'$n(x,0,0) \; \left[\mu m^{-3}\right]$', fontsize=14)

            axs[1].plot(self.y, self.n_array[self.num_grid_points[0]//2, :, self.num_grid_points[2]//2], c='k', marker='o', label=r'$n_{total}$')
            axs[1].set_title(r'$n(0,y,0)$', fontsize=18)
            axs[1].set_xlabel(r'$y \; \left[\mu m\right]$', fontsize=14)
            axs[1].set_ylabel(r'$n(0,y,0) \; \left[\mu m^{-3}\right]$', fontsize=14)

            axs[2].plot(self.z, self.n_array[self.num_grid_points[0]//2, self.num_grid_points[1]//2, :], c='k', marker='o', label=r'$n_{total}$')
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


    def plot_density_2d(self, **kwargs):
        """Plot the spatial density n(x,y,0), n(x,0,z) and n(0,y,z) along two directions respectively."""
        if (self.particle_props.T.value < 1e-3 and self.N_particles is not None) or np.linalg.norm(self.n_ex_array) > 0:
            title = kwargs.get('title', 'Spatial density, T='+str(self.particle_props.T)+', N='+str(int(self.N_particles)))
            filename = kwargs.get('filename', None)

            # Define the figure and GridSpec layout
            fig = plt.figure(figsize=(17, 5))  # Adjusted figure size for 1 row
            gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])  # 3 columns for plots, 1 for colorbars
            gs.update(wspace=0.65)  # Adjust spacing if needed

            # Create subplots
            axs = [plt.subplot(gs[0, i]) for i in range(3)]

            fig.suptitle(title, fontsize=24, y=0.94)

            # Plotting n(x,y,0), n(x,0,z), n(0,y,z)
            im = []
            im.append(axs[0].imshow(self.n_array[:, :, self.num_grid_points[2]//2].value, 
                                    extent=[self.x[0].value, self.x[-1].value, self.y[0].value, self.y[-1].value]))
            axs[0].set_title(r'$n(x,y,0)$', fontsize=18)

            im.append(axs[1].imshow(self.n_array[:, self.num_grid_points[1]//2, :].value, 
                                    extent=[self.x[0].value, self.x[-1].value, self.z[0].value, self.z[-1].value]))
            axs[1].set_title(r'$n(x,0,z)$', fontsize=18)

            im.append(axs[2].imshow(self.n_array[self.num_grid_points[0]//2, :, :].value, 
                                    extent=[self.y[0].value, self.y[-1].value, self.z[0].value, self.z[-1].value]))
            axs[2].set_title(r'$n(0,y,z)$', fontsize=18)

            # Set labels and colorbars for each plot
            for i, ax in enumerate(axs):
                ax.set_xlabel('x [μm]' if i != 2 else 'y [μm]', fontsize=12)
                ax.set_ylabel('y [μm]' if i != 2 else 'z [μm]', fontsize=12)

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


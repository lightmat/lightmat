import numpy as np
from scipy.integrate import simpson
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.units as u
from astropy.constants import hbar, k_B
from astropy.units import Quantity
from typing import Callable, Union


class BEC:
    """
    The BEC (Bose-Einstein Condensate) class is designed to model and analyze the properties of a Bose-Einstein 
    condensate. It allows for the calculation of the spacial density of weakly interacting bosons at low temperatures
    by iteratively updating the condensed density (either with Thomas Fermi approximation or by solving the 
    generalized Gross-Pitaevskii equation) and the thermal density (either with a semiclassical Hartree Fock or 
    semiclassical Popov approximation). It also fixes the chemical potential to ensure the normalization to a 
    specific particle number.

    Attributes:
        m (Quantity): Mass of a single particle in kilograms.
        a_s (Quantity): s-wave scattering length in meters.
        N_particles_target (int): Target number of particles in the condensate.
        T (Quantity): Temperature of the condensate in nanoKelvin.
        V_trap_array (Quantity): Array containing the external trapping potential values at each grid point.
        x, y, z (Quantity): 1D arrays of spatial grid points in micrometers.
        g (Quantity): Contact interaction strength.
        num_grid_points (tuple): Number of grid points in each spatial dimension.
        dx, dy, dz (Quantity): Grid spacings in each spatial dimension.
        X, Y, Z (np.ndarray): Meshgrid arrays for spatial coordinates.
        mu (Quantity): Chemical potential.
        n0_array, n_ex_array, n_array (np.ndarray): Arrays for the condensed density, non-condensed density, and total 
                                                    density respectively.
        N_particles, N_particles_condensed, N_particles_thermal (float): Total number of particles, number of condensed 
                                                                         particles, and number of thermal particles.
        condensate_fraction (float): Fraction of particles in the condensate state.
        use_TF, use_Popov (bool): Flags to indicate the use of Thomas-Fermi and Popov approximations respectively.
        convergence_history_mu, convergence_history_N (list): Lists to store the convergence history of the chemical 
                                                              potential and particle number.

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
            m: Union[float, Quantity], 
            a_s: Union[float, Quantity], 
            N_particles: int,
            T: Union[float, Quantity], 
            V_trap: Union[Callable, np.ndarray, Quantity], 
            x: Union[np.ndarray, Quantity] = np.linspace(-100, 100, 101) * u.um,
            y: Union[np.ndarray, Quantity] = np.linspace(-100, 100, 101) * u.um,
            z: Union[np.ndarray, Quantity] = np.linspace(-100, 100, 101) * u.um,
            init_with_zero_T: bool = True,
        ):
        """Initialize BEC class. Make sure to use the correct units as specified below!!! If you are 
           using astropy units, you can use units that are equilvalent to the ones specified below.
        
           Args:
               m: mass of a single particle in [kg]
               a_s: s-wave scattering length in [m]
               N_particles: number of particles
               T: temperature in [nK]
               V_trap: external trapping potential in [k_B x nK]. Can be either a callable that takes
                       x, y, z as arguments and returns the potential at each grid point, or a numpy array
                       of shape (len(x), len(y), len(z)) containing the potential at each grid point.
               x, y, z: 1d arrays of spacial grid points in [um]. If `V_trap` is an array, then `x`, `y` and `z`
                        must have corresponding shape. Defaults to np.linspace(-100, 100, 101) * u.um.
               init_with_zero_T: If True, run a zero-temperature calculation to improve initial guess for `mu`.
                                 If False, don't run a zero-temperature calculation (useful if you already know
                                 mu for some finite temperature, then you can set init_with_zero_T=False and
                                 provide the initial guess for mu yourself). Defaults to True.
        """
        # Atomic properties
        if isinstance(m, Quantity):
            self.m = m.to(u.kg)
        elif isinstance(m, (float, int)):
            self.m = m * u.kg
        else:
            raise TypeError("m must be either float or Quantity.")
        if isinstance(a_s, Quantity):
            self.a_s = a_s.to(u.m)
        elif isinstance(a_s, (float, int)):
            self.a_s = a_s * u.m
        else:
            raise TypeError("a_s must be either float or Quantity.")
        if isinstance(T, Quantity):
            self.T = T.to(u.nK)
        elif isinstance(T, (float, int)):
            self.T = T * u.nK
        else:
            raise TypeError("T must be either float or Quantity.")
        if isinstance(N_particles, int):
            self.N_particles_target = N_particles
        else:
            raise TypeError("N_particles must be int.")
        
        # Contact interaction strength 
        self.g = ((4*np.pi*(hbar**2)*self.a_s/self.m) / k_B).to(u.nK * u.um**3) 

        # Initialize the spatial grid 
        if isinstance(x, Quantity):
            if isinstance(x, np.ndarray):
                self.x = x.to(u.um)
            else:
                raise TypeError("If x is a Quantity, it must be a numpy array.")
        elif isinstance(x, np.ndarray):
            self.x = x * u.um
        else:
            raise TypeError("x must be either numpy array or Quantity.")
        if isinstance(y, Quantity):
            if isinstance(y, np.ndarray):
                self.y = y.to(u.um)
            else:
                raise TypeError("If y is a Quantity, it must be a numpy array.")
        elif isinstance(y, np.ndarray):
            self.y = y * u.um
        else:
            raise TypeError("y must be either numpy array or Quantity.")
        if isinstance(z, Quantity):
            if isinstance(z, np.ndarray):
                self.z = z.to(u.um)
            else:
                raise TypeError("If z is a Quantity, it must be a numpy array.")
        elif isinstance(z, np.ndarray):
            self.z = z * u.um
        else:
            raise TypeError("z must be either numpy array or Quantity.")
        self.num_grid_points = (len(self.x), len(self.y), len(self.z))
        self.dx = self.x[1] - self.x[0] # TODO: Generalize this to work for non-equidistant grids!
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij',)

        # Initialize external potential at each grid point
        if isinstance(V_trap, np.ndarray):
            if V_trap.shape == self.num_grid_points:
                if isinstance(V_trap, Quantity):
                    self.V_trap_array = V_trap.to(u.nK)
                elif isinstance(V_trap[0,0,0], (float, int)):
                    self.V_trap_array = V_trap * u.nK
                else:
                    raise TypeError("I V_trap is an array its entries must be float or Quantity in unit equivalent to [nK].")
            else:
                raise TypeError("If V_trap is an array, it must have same shape as num_grid_points. Please adapt space_size accordingly.")
        elif callable(V_trap):
            V = V_trap(self.X.value, self.Y.value, self.Z.value)    
            if isinstance(V, Quantity):
                self.V_trap_array = V.to(u.nK)
            elif isinstance(V[0,0,0], (float, int)):
                self.V_trap_array = V * u.nK
            else:
                raise TypeError("V_trap must either return float or Quantity in unit equivalent to [nK].")
        else:
            raise TypeError("V_trap must be either callable or numpy array.")
        
        # Initilize the chemical potential using the TF approximation. In principle this would give:
        # mu(r) = V(r) + g*(n0(r) + 2*n_ex(r))
        # But we don't want a position-dependent mu, so as an initial guess we take:
        self.mu = np.min(self.V_trap_array) + self.g.value*(self.N_particles_target**(1/3))*u.nK 

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
        if init_with_zero_T and self.T.value > 1e-3:
            T = self.T
            self.T = 0 * u.nK 
            self.eval_density(use_TF=True, use_Popov=False, show_progress=False)       
            self.T = T
        

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
            if self.T.value > 1e-3:
                self._update_n_ex(num_q_values) # This uses either the semiclassical HF or Popov approximation
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
            new_mu_direction = (self.N_particles_target - self.N_particles) / self.N_particles_target * u.nK 
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
               np.abs(self.N_particles_target-self.N_particles) < N_convergence_threshold*self.N_particles_target:
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
        self.n0_array = np.maximum((self.mu - (self.V_trap_array + 2*self.g*self.n_ex_array)) / self.g, 0)


    def _update_n0_by_solving_generalized_GPE(self,):
        """Solve the generalized GPE to update the condensed density `n0_array`."""
        raise NotImplementedError("This method is not implemented yet.")


    def _update_n_ex(
            self,
            num_q_values: int = 50,
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
        p_cutoff = np.sqrt(2*np.pi*self.m*k_B*self.T).to(u.u*u.m/u.s) # use this momentum units to deal with numerical
                                                                      # numbers roughly on the order ~1

        # Also, for increased numerical stability, we can rescale the integral to the interval 
        # [0,1] and integrate over the dimensionless variable q = p/p_cutoff instead of p.

        q_values = np.linspace(0, 1, num_q_values) 
        q_values = q_values[:, np.newaxis, np.newaxis, np.newaxis] # reshape to broadcast with spacial grid later 

        # Integrate using Simpson's rule (I chose this over quad() because quad() only works for scalar
        # integrands, but we have a 3d array of integrand values. So to use quad() we would need to loop
        # over the spatial grid and call quad() for each grid point, which is very slow. Simpson() can
        # integrate over the whole array at once in a vectorized way, which is much faster.)
        if self.use_Popov:
            integral = simpson(self._integrand_Popov(q_values, p_cutoff), q_values.flatten(), axis=0)
        else:
            integral = simpson(self._integrand_HF(q_values, p_cutoff), q_values.flatten(), axis=0)

        # Update n_ex_array 
        self.n_ex_array = np.maximum((integral*(p_cutoff.unit)**3 / (2*np.pi * hbar)**3).to(1/u.um**3), 0)


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
        eps_p = (p**2/(2*self.m) / k_B).to(u.nK) + self.V_trap_array + 2*self.g*(self.n0_array + self.n_ex_array)

        return p_cutoff.value*4*np.pi*p.value**2 / (np.exp((eps_p-self.mu) / self.T) - 1) 


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
        eps_p = np.sqrt(np.maximum(((p**2/(2*self.m) / k_B).to(u.nK) + self.V_trap_array + \
                        2*self.g*(self.n0_array + self.n_ex_array) - self.mu)**2 - (self.g*self.n0_array)**2, 0))

        num_non_condensed = ((p**2/(2*self.m) / k_B).to(u.nK) + self.V_trap_array + \
                              2*self.g*(self.n0_array + self.n_ex_array) - self.mu) / eps_p
        
        return p_cutoff.value*4*np.pi*p.value**2 * num_non_condensed / (np.exp(eps_p / self.T) - 1)


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
        if (self.T.value < 1e-3 and self.N_particles is not None) or np.linalg.norm(self.n_ex_array) > 0:
            title = kwargs.get('title', 'Convergence history, T='+str(self.T)+', N='+str(int(self.N_particles)))  
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
        if (self.T.value < 1e-3 and self.N_particles is not None) or np.linalg.norm(self.n_ex_array) > 0:
            title = kwargs.get('title', 'Spacial density, T='+str(self.T)+', N='+str(int(self.N_particles)))  
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


    def plot_density_2d(
            self, 
            **kwargs,
        ):
        """Plot the spacial density n(x,y,0), n(x,0,z) and n(0,y,z) along two directions respectively."""
        if (self.T.value < 1e-3 and self.N_particles is not None) or np.linalg.norm(self.n_ex_array) > 0:
            title = kwargs.get('title', 'Spacial density, T='+str(self.T)+', N='+str(int(self.N_particles)))
            filename = kwargs.get('filename', None) 

            # Define the figure and GridSpec layout
            fig = plt.figure(figsize=(17, 13))
            gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1, 0.05])  # 3 columns for plots, 1 for colorbars
            gs.update(wspace=0.65)  # Adjust spacing if needed

            # Create subplots
            axs = [[None for _ in range(3)] for _ in range(3)]
            for i in range(3):
                for j in range(3):
                    axs[i][j] = plt.subplot(gs[i, j])

            fig.suptitle(title, fontsize=24, y=0.94)

            im = [[None for _ in range(3)] for _ in range(3)]

            im[0][0] = axs[0][0].imshow(self.n_array[:,:,self.num_grid_points[2]//2].value, \
                                        extent=[self.x[0].value, self.x[-1].value, self.y[0].value, self.y[-1].value])
            axs[0][0].set_title(r'$n(x,y,0)$', fontsize=18)
            im[0][1] = axs[0][1].imshow(self.n_array[:,self.num_grid_points[1]//2,:].value, \
                                        extent=[self.x[0].value, self.x[-1].value, self.z[0].value, self.z[-1].value])
            axs[0][1].set_title(r'$n(x,0,z)$', fontsize=18)
            im[0][2] = axs[0][2].imshow(self.n_array[self.num_grid_points[0]//2,:,:].value, \
                                        extent=[self.y[0].value, self.y[-1].value, self.z[0].value, self.z[-1].value])
            axs[0][2].set_title(r'$n(0,y,z)$', fontsize=18)


            im[1][0] = axs[1][0].imshow(self.n0_array[:,:,self.num_grid_points[2]//2].value, \
                                        extent=[self.x[0].value, self.x[-1].value, self.y[0].value, self.y[-1].value])
            axs[1][0].set_title(r'$n_0(x,y,0)$', fontsize=18)
            im[1][1] = axs[1][1].imshow(self.n0_array[:,self.num_grid_points[1]//2,:].value, \
                                        extent=[self.x[0].value, self.x[-1].value, self.z[0].value, self.z[-1].value])
            axs[1][1].set_title(r'$n_0(x,0,z)$', fontsize=18)
            im[1][2] = axs[1][2].imshow(self.n0_array[self.num_grid_points[0]//2,:,:].value, \
                                        extent=[self.y[0].value, self.y[-1].value, self.z[0].value, self.z[-1].value])
            axs[1][2].set_title(r'$n_0(0,y,z)$', fontsize=18)


            im[2][0] = axs[2][0].imshow(self.n_ex_array[:,:,self.num_grid_points[2]//2].value, \
                                        extent=[self.x[0].value, self.x[-1].value, self.y[0].value, self.y[-1].value])
            axs[2][0].set_title(r'$n_{ex}(x,y,0)$', fontsize=18)
            im[2][1] = axs[2][1].imshow(self.n_ex_array[:,self.num_grid_points[1]//2,:].value, \
                                        extent=[self.x[0].value, self.x[-1].value, self.z[0].value, self.z[-1].value])
            axs[2][1].set_title(r'$n_{ex}(x,0,z)$', fontsize=18)
            im[2][2] = axs[2][2].imshow(self.n_ex_array[self.num_grid_points[0]//2,:,:].value, \
                                        extent=[self.y[0].value, self.y[-1].value, self.z[0].value, self.z[-1].value])
            axs[2][2].set_title(r'$n_{ex}(0,y,z)$', fontsize=18)


            for i in range(3):
                for j in range(3):
                    axs[i][0].set_xlabel('x [μm]', fontsize=12)
                    axs[i][0].set_ylabel('y [μm]', fontsize=12)
                    axs[i][1].set_xlabel('x [μm]', fontsize=12)
                    axs[i][1].set_ylabel('z [μm]', fontsize=12)
                    axs[i][2].set_xlabel('y [μm]', fontsize=12)
                    axs[i][2].set_ylabel('z [μm]', fontsize=12)

                    divider = make_axes_locatable(axs[i][j])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = fig.colorbar(im[i][j], cax=cax)
                    if i == 0:
                        cbar.ax.set_title(r'$n \; \left[ \mu m^{-3} \right]$', pad=6, fontsize=12, loc='left')
                    if i == 1:
                        cbar.ax.set_title(r'$n_0 \; \left[ \mu m^{-3} \right]$', pad=6, fontsize=12, loc='left')
                    if i == 2:
                        cbar.ax.set_title(r'$n_{ex} \; \left[ \mu m^{-3} \right]$', pad=6, fontsize=12, loc='left')
                    
            if filename != None:
                fig.savefig(filename, dpi=300, bbox_inches='tight')

            return axs
        
        else:
            print("No convergence history found. Please run eval_density() first.")
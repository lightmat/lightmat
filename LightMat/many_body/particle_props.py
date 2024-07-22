import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.units as u
from astropy.constants import hbar, k_B
from astropy.units import Quantity
from typing import Callable, Union, Sequence
import copy



class ParticleProps:
    """
    A class representing the properties of a particle system.

    This class encapsulates the properties and behaviors of a particle system in a 
    quantum simulation context. It supports defining the mass, temperature, domain, and
    trapping potential of particles, as well as plotting the trapping potential.

    Attributes:
        name (str): Name of the particle.
        species (str): Type of the particle, either 'fermion' or 'boson'.
        m (Quantity or float): Mass of the particle, in kilograms.
        N_particles (int): Number of particles in the system.
        T (Quantity or float): Temperature of the system, in nanoKelvins.
        domain (Sequence, np.ndarray, or Quantity): Spatial domain of the system.
        a_s (Quantity or float, optional): s-wave scattering length of the particles.
        g (Quantity): Contact interaction strength calculated from a_s.

    Methods:
        V_trap(x,y,z): Returns the trapping potential of the system at given position(s).
        plot_V_trap(x,y,z): Plots the trapping potential of the system at given position(s).
    """
    def __init__(
            self, 
            species: str, 
            m: Union[float, Quantity], 
            N_particles: int, 
            T: Union[float, Quantity],
            domain: Union[Sequence[float], Sequence[Sequence[float]], np.ndarray, Quantity],
            V_trap_func: Union[Callable, None],
            V_trap_array: Union[np.ndarray, Quantity, None] = None, 
            a_s: Union[float, Quantity, None] = None,
            name: str = "Particle",
            **V_trap_func_kwargs,
        ):
        """Initializes the ParticleProps class. Either provide a callable V_trap_func or a 3d array V_trap_array for the 
           potential.
        
           Args:
               species: Type of the particle, either 'fermion' or 'boson'.
               m: Mass of the particle, in [kg].
               N_particles: Number of particles in the system.
               T: Temperature of the atomic gas, in [nK].
               domain: Spatial domain of the system. Either a sequence of length 2 containing the same x,y,z domain, 
                       or a sequence of length 3 containing sequences of length 2 containing the x,y,z domain in [um].
               V_trap_func: Function that returns the trapping potential of the system at given position(s) in [kB x nK]. 
                            This will be used if V_trap_array is not provided. Defaults to None.
               V_trap_array: 3d array containing the trapping potential of the system at given position(s) in [kB x nK]. 
                             This will be used if no V_trap_func is provided. Defaults to None.
               a_s (Quantity or float, optional): s-wave scattering length of the particles in [m].
               name (str): Name of the particle.
               **V_trap_func_kwargs: Keyword arguments to pass to V_trap_func.
        """
        #Name
        if isinstance(name, str):
            self.name = name
        else:
            raise TypeError("name must be a string.")
        
        # Species
        if isinstance(species, str):
            if species not in ["fermion", "boson"]:
                raise ValueError("species must be either 'fermion' or 'boson'.")
            self.species = species
        else:
            raise TypeError("species must be a string, either 'fermion' or 'boson'.")

        # Mass
        if isinstance(m, Quantity) and m.unit.is_equivalent(u.kg):
            self.m = m.to(u.kg)
        elif isinstance(m, (float, int)):
            self.m = m * u.kg
        else:
            raise TypeError("m must be either a float, int, or Quantity in unit equivalent to kg.")

        # Number of particles
        if isinstance(N_particles, int):
            self.N_particles = N_particles
        else:
            raise TypeError("N_particles must be an integer.")

        # Temperature
        if isinstance(T, Quantity) and T.unit.is_equivalent(u.nK):
            self.T = T.to(u.nK)
        elif isinstance(T, (float, int)):
            self.T = T * u.nK
        else:
            raise TypeError("T must be either a float, int, or Quantity in unit equivalent to nK.")
        
        # Domain
        if isinstance(domain, Quantity) and domain.unit.is_equivalent(u.um):
                domain = domain.to(u.um).value
        else:
            raise u.UnitsError("If domain is a Quantity, it must be in units equivalent to um.")
        if isinstance(domain, (Sequence, np.ndarray)):
            if len(domain) == 2:
                if all(isinstance(d, (float, int)) for d in domain):
                    self.domain = np.array([np.asarray(domain), np.asarray(domain), \
                                            np.asarray(domain)]) * u.um
                else:
                    raise TypeError("If domain is a sequence of length 2, it must contain floats.")
            elif len(domain) == 3:
                if all(isinstance(d, (Sequence, np.ndarray)) for d in domain):
                    if all(len(d) == 2 for d in domain):
                        if all(isinstance(d_i, (float, int)) for d in domain for d_i in d):
                            self.domain = np.array([np.asarray(d) for d in domain]) * u.um
                        else:
                            raise TypeError("If domain is a sequence of length 3, it must contain sequences of length 2 "
                                            "containing floats.")
                    else:
                        raise TypeError("If domain is a sequence of length 3, it must contain sequences of length 2 "
                                        "containing floats.")
                else:
                    raise TypeError("If domain is a sequence of length 3, it must contain sequences of length 2 "
                                    "containing floats.")
            else:
                raise TypeError("domain must be a sequence of length 2 or or a sequence of length 3 containing "
                                "sequences of length 2")
        else:
            raise TypeError("domain must be a sequence of length 2 or or a sequence of length 3 containing "
                            "sequences of length 2")
        
        
        # Trap potential
        if isinstance(V_trap_func, Callable):
            self._V_trap_func = V_trap_func
            self._V_trap_func_kwargs = V_trap_func_kwargs
            self.V_trap_func.__func__.__doc__ = V_trap_func.__doc__
        elif isinstance(V_trap_array, Quantity) and V_trap_array.unit.is_equivalent(u.nK):
            pass
        elif V_trap_func is not None:
            raise TypeError("V_trap_func must be a callable or None.")

        # s-wave scattering length
        if self.species == "fermion":
            if a_s is not None:
                print(f'Fermions do not interact via s-wave scattering, the s-wave sacttering length of {self.name} is set to None.')
            self.a_s = None
        else:
            if isinstance(a_s, Quantity) and a_s.unit.is_equivalent(u.m):
                self.a_s = a_s.to(u.m)
            elif isinstance(a_s, (float, int)):
                self.a_s = a_s * u.m
            elif a_s is None:
                self.a_s = None
                print(f'You provided no s-wave scattering length for {self.name}, all calculations will assume no interactions!')
            else:
                raise TypeError("a_s must be either a float, int, or Quantity.")
            
        # Contact interaction strength
        if self.a_s is None:
            self.g = None
        else:
            self.g = ((4*np.pi*(hbar**2)*self.a_s/self.m) / k_B).to(u.nK * u.um**3) 

        # V_trap_func_kwargs
        for key, value in V_trap_func_kwargs.items():
            setattr(self, key, value)


    def V_trap_func(self, *args):
        """
        Calculates the trap potential at given position(s).

        This method acts as a wrapper around the trap potential function ('V_trap_func') 
        passed during instantiation. 

        Parameters:
            *args: Variable length argument list representing position(s).

        Returns:
            The value of the trap potential at the given position(s).
        """
        return self._V_trap_func(*args, **self._V_trap_func_kwargs)
    

    def print_props(self,):
        """Prints the properties of the particle system."""
        print(f"Name: {self.name}")
        print(f"Species: {self.species}")
        print(f"Mass: {self.m}")
        print(f"Number of particles: {self.N_particles}")
        print(f"Temperature: {self.T}")
        print(f"Domain: {self.domain}")
        if self.a_s is not None:
            print(f"s-wave scattering length: {self.a_s}")
        if self.g is not None:
            print(f"Contact interaction strength: {self.g}")
        print(f"Trap potential: {self._V_trap_func.__name__}")


    def plot_V_trap(
            self, 
            which: str = "all",
            num_grid_points: int = 201,
            **kwargs,
        ):
        """Plots the trap potential of the system.
        
           Args:
               which (str): Which variable to plot the trap potential as a function of. Must be one of 'all',
               'all1d', 'all2d', 'x', 'y', 'z', 'xy', 'xz', or 'yz'. Defaults to 'all'.
        """
        if which not in ['all', 'all1d', 'all2d', 'x', 'y', 'z', 'xy', 'xz', 'yz']:
            raise ValueError("which must be one of 'all', 'all1d', 'all2d', 'x', 'y', 'z', 'xy', 'xz', or 'yz'.")
        
        title = kwargs.get('title', "Trap potential of " + self.name)  
        filename = kwargs.get('filename', None)
        
        x = np.linspace(self.domain[0,0].value, self.domain[0,1].value, num_grid_points) 
        y = np.linspace(self.domain[1,0].value, self.domain[1,1].value, num_grid_points) 
        z = np.linspace(self.domain[2,0].value, self.domain[2,1].value, num_grid_points) 
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        V_trap_result = self.V_trap_func(X, Y, Z)
        if isinstance(V_trap_result, Quantity):
            V_trap_result = V_trap_result.to(u.nK).value

        if which in ['all', 'all1d', 'all2d']:
            if which == 'all1d':
                settings = ['x', 'y', 'z']
                nrows, ncols = 1, 3
                figsize = (15, 5)  # Wider figure for 1D plots
                titles = ['Along X', 'Along Y', 'Along Z']
            elif which == 'all2d':
                settings = ['xy', 'xz', 'yz']
                nrows, ncols = 1, 3
                figsize = (15, 5)  # Wider figure for 2D plots
                titles = ['XY Plane', 'XZ Plane', 'YZ Plane']
            elif which == 'all':
                settings = ['x', 'y', 'z', 'xy', 'xz', 'yz']
                nrows, ncols = 2, 3
                figsize = (15, 10)  # Larger figure for both 1D and 2D plots
                titles = ['Along X', 'Along Y', 'Along Z', 'XY Plane', 'XZ Plane', 'YZ Plane']

            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            fig.suptitle(title, fontsize=22)

            for ax, setting, title in zip(axs.flatten(), settings, titles):
                ax.set_title(title, fontsize=18)
                if len(setting) == 1:
                    self._plot_1d(V_trap_result, num_grid_points, fig, ax, setting)
                elif len(setting) == 2:
                    self._plot_2d(V_trap_result, x, y, z, num_grid_points, fig, ax, setting)

            plt.tight_layout()
            plt.show()

        else:
            if len(which) == 1:
                self._plot_1d(None, None, which)
            elif len(which) == 2:
                self._plot_2d(None, None, which)

        if filename is not None:
            fig.savefig(filename, dpi=300, bbox_inches='tight')


    def _plot_1d(
            self, 
            V_trap_result,
            num_grid_points,
            fig, 
            ax, 
            which,
        ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.suptitle("Trap potential of " + self.name, fontsize=22)

        if which == "x":
            x = np.linspace(self.domain[0,0].value, self.domain[0,1].value, num_grid_points) 
            ax.plot(x, V_trap_result[:, num_grid_points//2, num_grid_points//2])
            ax.set_xlabel(r"$x \; [\mu m]$", fontsize=18)
            ax.set_ylabel(r"$V_{trap}(x, 0, 0) \; [nK]$", fontsize=18)
        elif which == "y":
            y = np.linspace(self.domain[1,0].value, self.domain[1,1].value, num_grid_points) 
            ax.plot(y, V_trap_result[num_grid_points//2, :, num_grid_points//2])
            ax.set_xlabel(r"$y \; [\mu m]$", fontsize=18)
            ax.set_ylabel(r"$V_{trap}(0, y, 0) \; [nK]$", fontsize=18)
        elif which == "z":
            z = np.linspace(self.domain[2,0].value, self.domain[2,1].value, num_grid_points)
            ax.plot(z, V_trap_result[num_grid_points//2, num_grid_points//2, :])
            ax.set_xlabel(r"$z \; [\mu m]$", fontsize=18)
            ax.set_ylabel(r"$V_{trap}(0, 0, z) \; [nK]$", fontsize=18)

        ax.grid(True)

    def _plot_2d(
            self, 
            V_trap_result,
            x, 
            y, 
            z,
            num_grid_points,
            fig, 
            ax, 
            which,
        ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.suptitle("Trap potential of " + self.name, fontsize=22)

        im = None
        if which == "xy":
            im = ax.pcolormesh(x, y, V_trap_result[:,:,num_grid_points//2].T)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel(r"$x \; [\mu m]$", fontsize=18)
            ax.set_ylabel(r"$y \; [\mu m]$", fontsize=18)
        elif which == "xz":
            im = ax.pcolormesh(x, z, V_trap_result[:,num_grid_points//2,:].T)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel(r"$x \; [\mu m]$", fontsize=18)
            ax.set_ylabel(r"$z \; [\mu m]$", fontsize=18)
        elif which == "yz":
            im = ax.pcolormesh(y, z, V_trap_result[num_grid_points//2,:,:].T)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel(r"$y \; [\mu m]$", fontsize=18)
            ax.set_ylabel(r"$z \; [\mu m]$", fontsize=18)

        if im is not None:
            # Create an axis for the colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(r'$V_{\mathrm{trap}} \; [nK]$', fontsize=18)


    def copy(self):
        """Create a shallow copy of the ParticleProps instance."""
        return copy.copy(self)

    def deepcopy(self):
        """Create a deep copy of the ParticleProps instance."""
        return copy.deepcopy(self)
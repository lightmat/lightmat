import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import hbar, k_B
from astropy.units import Quantity
from typing import Callable, Union, Sequence



class ParticleProps:
    def __init__(
            self, 
            name: str,
            species: str, 
            m: Union[float, Quantity], 
            N_particles: int, 
            T: Union[float, Quantity],
            domain: Union[Sequence[float], Sequence[Sequence[float]], np.ndarray, Quantity],
            V_trap: Union[Callable, np.ndarray, Quantity], 
            a_s: Union[float, Quantity, None] = None,
            **V_trap_kwargs,
        ):
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
        if isinstance(V_trap, Callable):
            self._V_trap = V_trap
            self._V_trap_kwargs = V_trap_kwargs
            self.V_trap.__func__.__doc__ = V_trap.__doc__
        elif isinstance(V_trap, np.ndarray):
            raise NotImplementedError("V_trap must be a callable.")

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


    def V_trap(self, *args):
        """
        This is a placeholder docstring. It will be replaced by the docstring of V_trap.
        """
        return self._V_trap(*args, **self._V_trap_kwargs)


    def plot_V_trap(
        self,
        x: Union[float, Sequence[float], np.ndarray, Quantity], 
        y: Union[float, Sequence[float], np.ndarray, Quantity],
        z: Union[float, Sequence[float], np.ndarray, Quantity],
    ):
        # Check input
        if isinstance(x, Quantity) and x.unit.is_equivalent(u.um):
            x = x.to(u.um).value
        if isinstance(x, (float, int)):
            x = np.array([x])
        elif isinstance(x, (Sequence, np.ndarray)):
            x = np.array(x)
        else:
            raise TypeError("x must be a float, int, Sequence, or Quantity.")

        if isinstance(y, Quantity) and y.unit.is_equivalent(u.um):
            y = y.to(u.um).value
        if isinstance(y, (float, int)):
            y = np.array([y])
        elif isinstance(y, (Sequence, np.ndarray)):
            y = np.array(y)
        else:
            raise TypeError("y must be a float, int, Sequence, or Quantity.")
        
        if isinstance(z, Quantity) and z.unit.is_equivalent(u.um):
            z = z.to(u.um).value
        if isinstance(z, (float, int)):
            z = np.array([z])
        elif isinstance(z, (Sequence, np.ndarray)):
            z = np.array(z)
        else:
            raise TypeError("z must be a float, int, Sequence, or Quantity.")

        plot_dim = sum(len(arr) > 1 for arr in [x, y, z])
        if plot_dim == 0:
            raise ValueError("At least one of x, y, or z must be a sequence of length > 1.")
        elif plot_dim == 1:
            self._plot_1d(x, y, z)
        elif plot_dim == 2:
            self._plot_2d(x, y, z)
        elif plot_dim == 3:
            raise ValueError("Cannot plot a 3D function, at least one of x,y,z have to be a number, not a sequence.")      
        

    def _plot_1d(
        self,
        x, 
        y,
        z,
    ):
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle("Trap potential of " + self.name, fontsize=22)
        ax.grid(True)
        if len(x) > 1:
            ax.plot(x, self.V_trap(x, y, z))
            ax.set_xlabel(r"$x \; [\mu m]$", fontsize=18)
            ax.set_ylabel(r"$V_{trap}$(x, " + str(y[0]) + ", " + str(z[0]) + r") $[nK]$", fontsize=18)
        elif len(y) > 1:
            ax.plot(y, self.V_trap(x, y, z))
            ax.set_xlabel(r"$y \; [\mu m]$", fontsize=18)
            ax.set_ylabel(r"$V_{trap}$(" + str(x[0]) + ", y, " + str(z[0]) + r") $[nK]$", fontsize=18)
        elif len(z) > 1:
            ax.plot(z, self.V_trap(x, y, z))
            ax.set_xlabel(r"$z \; [\mu m]$", fontsize=18)
            ax.set_ylabel(r"$V_{trap}$(" + str(x[0]) + ", " + str(y[0]) + ", z)" + r" $[nK]$", fontsize=18)

        plt.show()


    def _plot_2d(
            self,
            x,
            y,
            z,
        ):
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        # Assuming V_trap returns an array of shape (len(x), len(y), len(z))
        V_trap_result = self.V_trap(X, Y, Z)
        if isinstance(V_trap_result, Quantity):
            V_trap_result = V_trap_result.to(u.nK).value

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle("Trap potential of " + self.name, fontsize=22)
        im = None
        if len(x) > 1 and len(y) > 1:
            im = ax.imshow(V_trap_result[:, :, 0], extent=[x[0], x[-1], y[0], y[-1]])
            ax.set_xlabel(r"$x \; [\mu m]$", fontsize=18)
            ax.set_ylabel(r"$y \; [\mu m]$", fontsize=18)
        elif len(x) > 1 and len(z) > 1:
            im = ax.imshow(V_trap_result[:, 0, :], extent=[x[0], x[-1], z[0], z[-1]])
            ax.set_xlabel(r"$x \; [\mu m]$", fontsize=18)
            ax.set_ylabel(r"$z \; [\mu m]$", fontsize=18)
        elif len(y) > 1 and len(z) > 1:
            im = ax.imshow(V_trap_result[0, :, :], extent=[y[0], y[-1], z[0], z[-1]])
            ax.set_xlabel(r"$y \; [\mu m]$", fontsize=18)
            ax.set_ylabel(r"$z \; [\mu m]$", fontsize=18)

        # Add a colorbar with a label
        if im is not None:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(r'$V_{\mathrm{trap}}$ [nK]', fontsize=18)

        plt.show()
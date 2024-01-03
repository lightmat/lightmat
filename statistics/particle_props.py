import numpy as np
from scipy.integrate import simpson
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.units as u
from astropy.constants import hbar, k_B
from astropy.units import Quantity
from typing import Callable, Union, Sequence, Tuple, Dict



class ParticleProps:
    def __init__(
            self, 
            name: str,
            species: str, 
            m: Union[float, Quantity], 
            N_particles: int, 
            T: Union[float, Quantity],
            V_trap: Union[Callable, np.ndarray, Quantity], 
            a_s: Union[float, Quantity, None] = None,
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
        if isinstance(m, Quantity):
            self.m = m.to(u.kg)
        elif isinstance(m, (float, int)):
            self.m = m * u.kg
        else:
            raise TypeError("m must be either a float, int, or Quantity.")

        # Number of particles
        self.N_particles = N_particles

        # Temperature
        if isinstance(T, Quantity):
            self.T = T.to(u.nK)
        elif isinstance(T, (float, int)):
            self.T = T * u.nK
        else:
            raise TypeError("T must be either a float, int, or Quantity.")

        # s-wave scattering length
        if self.species == "fermion":
            if a_s is not None:
                print(f'Fermions do not interact via s-wave scattering, the s-wave sacttering length of {self.name} is set to None.')
            self.a_s = None
        else:
            if isinstance(a_s, Quantity):
                self.a_s = a_s.to(u.m)
            elif isinstance(a_s, (float, int)):
                self.a_s = a_s * u.m
            elif a_s is None:
                self.a_s = None
                print(f'You provided no s-wave scattering length for {self.name}, all calculations will assume no interactions!')
            else:
                raise TypeError("a_s must be either a float, int, or Quantity.")
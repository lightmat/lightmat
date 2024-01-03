import numpy as np
import astropy.units as u
from astropy.constants import hbar, k_B
from astropy.units import Quantity
from typing import Union, Sequence, Tuple, Dict

from .particle_props import ParticleProps



class DensityCalculator():
    def __init__(
            self, 
            particle_props: Union[ParticleProps, Sequence[ParticleProps], np.ndarray], 
            a_s_interspecies: Union[Dict[Tuple[str, str], Union[float, Quantity]], None] = None
        ):
        # Particle properties
        if isinstance(particle_props, ParticleProps):
            self.particle_props_list = [particle_props]
        elif isinstance(particle_props, (Sequence, np.ndarray)):
            if any(not isinstance(p, ParticleProps) for p in particle_props):
                raise TypeError("particle_props must be a ParticleProps or Sequence of ParticleProps.")
            self.particle_props_list = list(particle_props)
        else:
            raise TypeError("particle_props must be a ParticleProps or Sequence of ParticleProps.")

        # Ensure unique names for each particle type
        if len({p.name for p in self.particle_props_list}) != len(self.particle_props_list):
            raise ValueError("Each particle kind must have a unique name.")

        # Interspecies scattering lengths    
        if a_s_interspecies == None: 
            if len(self.particle_props_list) > 1 and any(p.species == "boson" for p in self.particle_props_list): 
                print("Warning: You provided no interspecies s-wave scattering length, but you have multiple species of particles, at least one of which is a boson. All calculations will assume no inter-species interactions!")
            self.a_s_interspecies = None
        elif isinstance(a_s_interspecies, dict):
            self.a_s_interspecies = {}
            for (name1, name2), a_s_value in a_s_interspecies.items():
                if name1 not in {p.name for p in self.particle_props_list}:
                    raise ValueError(f"You provided an interspecies s-wave scattering length between particles with name {name1} and {name2}, but did not provide `particle_props` for {name1}.")
                if name2 not in {p.name for p in self.particle_props_list}:
                    raise ValueError(f"You provided an interspecies s-wave scattering length between particles with name {name1} and {name2}, but did not provide `particle_props` for {name2}.")
                if name1 == name2:
                    raise ValueError(f"You provided an interspecies s-wave scattering length between particles with name {name1} and {name2}, but they are the same particle!")
                
                if isinstance(a_s_value, Quantity):
                    self.a_s_interspecies[(name1, name2)] = a_s_value.to(u.m)
                elif isinstance(a_s_value, (float, int)):
                    self.a_s_interspecies[(name1, name2)] = a_s_value * u.m
                else:
                    raise TypeError(f"interspecies s-wave scattering length between particles with name {name1} and {name2} must be either a float or Quantity.")
        else:
            raise TypeError("interspecies s-wave scattering lengths must be a dictionary with key (name1,name2) of particle names and value of type float or Quantity.")
        

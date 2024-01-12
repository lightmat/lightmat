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


class BoseFermiGas:
    def __init__(
            self, 
            bose_props: ParticleProps, 
            fermi_props: ParticleProps,
    ):
        # Initialize the particle properties
        if isinstance(bose_props, ParticleProps):
            if bose_props.species == 'boson':
                self.bose_props = bose_props
            else:
                raise ValueError('boson_props must be a boson ParticleProps object')
        else:
            raise TypeError('boson_props must be a ParticleProps object')
        
        if isinstance(fermi_props, ParticleProps):
            if fermi_props.species == 'fermion':
                self.fermi_props = fermi_props
            else:
                raise ValueError('fermion_props must be a fermion ParticleProps object')
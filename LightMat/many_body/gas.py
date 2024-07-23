from abc import ABC, abstractmethod
from astropy import units as u
from astropy.constants import c, eps0
import numpy as np
from typing import Union
from collections.abc import Sequence

from .particle_props import ParticleProps

class Gas(ABC):
    @abstractmethod
    def __init__(
        self,
        particle_props: ParticleProps,
        V_array_3d: Union[np.ndarray, u.Quantity],
        x: Union[Sequence[float], np.ndarray, u.Quantity],
        y: Union[Sequence[float], np.ndarray, u.Quantity],
        z: Union[Sequence[float], np.ndarray, u.Quantity],
    ) -> None:
        """
        Initializes the abstract Gas() class.

        Args:
            particle_props: ParticleProps() object containing the properties of the atomic particles.
            V_array_3d: External potential the atoms feel in [kB x nK] as 3d array of shape (len(x), len(y), len(z)).
            x: Sequence of x-coordinates in [um] corresponding to the values of the V_array along axis=0.
            y: Sequence of y-coordinates in [um] corresponding to the values of the V_array along axis=1.
            z: Sequence of z-coordinates in [um] corresponding to the values of the V_array along axis=2.
        """
        self.particle_props = particle_props
        self.V_array_3d = V_array_3d
        self.x = x
        self.y = y
        self.z = z
        
    

    @abstractmethod
    def eval_density(
            self,
        ) -> None:
        """Run the iterative procedure to update in turn the densities spatial density of the gas and store it in the
           self.n_array attribute.
        """
        pass
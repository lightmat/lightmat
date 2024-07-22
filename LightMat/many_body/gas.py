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
        V_array: Union[np.ndarray, u.Quantity],
        x: Union[Sequence[float], np.ndarray, u.Quantity],
        y: Union[Sequence[float], np.ndarray, u.Quantity],
        z: Union[Sequence[float], np.ndarray, u.Quantity],
    ) -> None:
        """
        Initializes the abstract Gas() class.

        Args:
            particle_props: ParticleProps() object containing the properties of the atomic particles.
            V_array: External potential the atoms feel in [kB x nK].
            x: Sequence of x-coordinates in [um] corresponding to the values of the V_array along axis=0.
            y: Sequence of y-coordinates in [um] corresponding to the values of the V_array along axis=1.
            z: Sequence of z-coordinates in [um] corresponding to the values of the V_array along axis=2.
        """
        self.particle_props = particle_props
        self.V_array = V_array
        self.x = x
        self.y = y
        self.z = z
        
    

    @abstractmethod
    def E_vec(
            self,
            x: Union[float, Sequence[float], np.ndarray, u.Quantity],
            y: Union[float, Sequence[float], np.ndarray, u.Quantity],
            z: Union[float, Sequence[float], np.ndarray, u.Quantity],
    ) -> u.Quantity:
        """Returns the complex electric field vector of the beam at the position (x,y,z) in [V/m] in the standard Carteesian coordinate system.
           This is just the complex electric field amplitude multiplied by the 3d polarization vector. Here, x, y, z are the global
           standard Carteesian coordinates in [um] and can be either float or array obtained from np.meshgrid().
           
           Args:
                x: Global standard Carteesian coordinate in [um].
                y: Global standard Carteesian coordinate in [um].
                z: Global standard Carteesian coordinate in [um].
                
           Returns:
                u.Quantity: Complex electric field vector of the beam at the position (x,y,z) in [V/m] in the standard Carteesian coordinate system.
        """
        pass


    @abstractmethod
    def E(
            self, 
            x: Union[float, Sequence[float], np.ndarray, u.Quantity], 
            y: Union[float, Sequence[float], np.ndarray, u.Quantity], 
            z: Union[float, Sequence[float], np.ndarray, u.Quantity],
    ) -> u.Quantity:
        """Returns the complex electric field amplitude of the laser at the position (x,y,z) in [V/m]. 
           Here, x, y, z are the global standard Carteesian coordinates in [um] and can be either float or array obtained from np.meshgrid().

           Args:
                x: Global standard Carteesian coordinate in [um].
                y: Global standard Carteesian coordinate in [um].
                z: Global standard Carteesian coordinate in [um].

           Returns:
                u.Quantity: Complex electric field amplitude of the beam at the position (x,y,z) in [V/m], can be either float or array.
        """
        pass

    @abstractmethod
    def I(
            self,
            x: Union[float, Sequence[float], np.ndarray, u.Quantity],
            y: Union[float, Sequence[float], np.ndarray, u.Quantity],
            z: Union[float, Sequence[float], np.ndarray, u.Quantity],
    ) -> u.Quantity:    
        """Returns the intensity of the laser at the position (x,y,z) in [mW/cm^2]. Here, x, y, z are the global standard Carteesian
           coordinates in [um] and can be either float or array obtained from np.meshgrid().

           Args:
                x: Global standard Carteesian coordinate in [um].
                y: Global standard Carteesian coordinate in [um].
                z: Global standard Carteesian coordinate in [um].

           Returns:
                u.Quantity: Intensity of the beam at the position (x,y,z) in [mW/cm^2], can be either float or array.
        """
        pass

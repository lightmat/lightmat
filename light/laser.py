from abc import ABC, abstractmethod
from astropy import units as u
from astropy.constants import c, eps0
import numpy as np
from typing import Union
from collections.abc import Sequence

class Laser(ABC):
    @abstractmethod
    def __init__(
        self,
    ) -> None:
        """
        Initializes the laser.
        """
        pass
    

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

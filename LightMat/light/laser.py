from abc import ABC, abstractmethod
from astropy import units as u
from astropy.constants import c, eps0
import numpy as np
from typing import Union
from collections.abc import Sequence

from .beams import Beam

class Laser(ABC):
    @abstractmethod
    def __init__(
        self,
        name: str,
        beams: Sequence[Beam],
    ) -> None:
        """
        Initializes the laser. A laser can have several beams. The beams interfere with each other to 
        generate the total electric field of the laser.

        Args:
            name: Name of the laser.
            beams: Sequence of beams in the laser.
            color: Color of the laser.
        """
        self.name = name
        self.beams = beams
    

    @abstractmethod
    def E_vec(
            self,
            x: Union[float, Sequence[float], np.ndarray, u.Quantity],
            y: Union[float, Sequence[float], np.ndarray, u.Quantity],
            z: Union[float, Sequence[float], np.ndarray, u.Quantity],
    ) -> u.Quantity:
        """Returns the total complex electric field vector of the laser at the position (x,y,z) in [V/m] in the standard Carteesian 
           coordinate system. All ``beams`` of the laser interfere with each other. Here, x, y, z are the global standard Carteesian 
           coordinates in [um] and can be either float or array obtained from np.meshgrid().
           
           Args:
                x: Global standard Carteesian coordinate in [um].
                y: Global standard Carteesian coordinate in [um].
                z: Global standard Carteesian coordinate in [um].
                
           Returns:
                u.Quantity: Complex electric field vector of the laser at the position (x,y,z) in [V/m] in the standard Carteesian coordinate system.
        """
        pass


    @abstractmethod
    def E(
            self, 
            x: Union[float, Sequence[float], np.ndarray, u.Quantity], 
            y: Union[float, Sequence[float], np.ndarray, u.Quantity], 
            z: Union[float, Sequence[float], np.ndarray, u.Quantity],
    ) -> u.Quantity:
        """Returns the complex electric field amplitude of the laser at the position (x,y,z) in [V/m]. All ``beams`` of the laser interfere 
           with each other. Here, x, y, z are the global standard Carteesian coordinates in [um] and can be either float or array obtained 
           from np.meshgrid().

           Args:
                x: Global standard Carteesian coordinate in [um].
                y: Global standard Carteesian coordinate in [um].
                z: Global standard Carteesian coordinate in [um].

           Returns:
                u.Quantity: Complex electric field amplitude of the laser at the position (x,y,z) in [V/m], can be either float or array.
        """
        pass

    @abstractmethod
    def I(
            self,
            x: Union[float, Sequence[float], np.ndarray, u.Quantity],
            y: Union[float, Sequence[float], np.ndarray, u.Quantity],
            z: Union[float, Sequence[float], np.ndarray, u.Quantity],
    ) -> u.Quantity:    
        """Returns the intensity of the laser at the position (x,y,z) in [mW/cm^2]. All ``beams`` of the laser interfere 
           with each other. Here, x, y, z are the global standard Carteesian coordinates in [um] and can be either float 
           or array obtained from np.meshgrid().

           Args:
                x: Global standard Carteesian coordinate in [um].
                y: Global standard Carteesian coordinate in [um].
                z: Global standard Carteesian coordinate in [um].

           Returns:
                u.Quantity: Intensity of the laser at the position (x,y,z) in [mW/cm^2], can be either float or array.
        """
        pass
